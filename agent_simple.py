from typing import Optional, TypedDict, Annotated
import json, sqlite3, requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_community.utilities.sql_database import SQLDatabase
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.runnables import RunnableLambda
from langgraph.types import interrupt
from typing_extensions import Literal
from contextvars import ContextVar
from langgraph.prebuilt import tools_condition

from langchain_anthropic import ChatAnthropic
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import Runnable, RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver


# ---------------- Environment / Model ----------------
load_dotenv()
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")
graph_state_ctx: ContextVar[dict] = ContextVar("graph_state_ctx", default={})


# ---------------- State ----------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_authed: bool
    customer_id: Optional[int]

# ---------------- Database ----------------
def get_engine_for_chinook_db():
    url = "https://raw.githubusercontent.com/lerocha/chinook-database/master/ChinookDatabase/DataSources/Chinook_Sqlite.sql"
    sql_script = requests.get(url).text
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.executescript(sql_script)
    return create_engine(
        "sqlite://",
        creator=lambda: conn,
        poolclass=StaticPool,
        connect_args={"check_same_thread": False},
    )

engine = get_engine_for_chinook_db()
db = SQLDatabase(engine)



# ---------------- Helpers ----------------


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\n please fix your mistakes.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }

def create_tool_node_with_fallback(tools: list) -> dict:
    return ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

def create_scoped_tool_node(tools: list):
    base = ToolNode(tools).with_fallbacks(
        [RunnableLambda(handle_tool_error)], exception_key="error"
    )

    def run_with_state(state):
        token = graph_state_ctx.set(state)  
        try:
            return base.invoke(state)
        finally:
            graph_state_ctx.reset(token)

    return RunnableLambda(run_with_state)


def enforce_customer_scope(customer_id: Optional[int]) -> Optional[dict]:
    st = graph_state_ctx.get({})
    if not st.get("is_authed"):
        return {"error": "AUTH_REQUIRED", "message": "Authenticate first."}

    authed_id = st.get("customer_id")
    if authed_id is None:
        return {"error": "AUTH_REQUIRED", "message": "No authenticated customer_id in state."}

    try:
        req_id = int(customer_id) if customer_id is not None else None
        authed_id = int(authed_id)
    except Exception:
        return {"error": "FORBIDDEN", "message": f"Invalid customer_id. Use authenticated customer_id={authed_id}."}

    if req_id is None:
        # You can choose to inject here instead; since this is inside the tool,
        # safest is to force caller to pass correct id explicitly.
        return {"error": "FORBIDDEN", "message": f"Missing customer_id. Use authenticated customer_id={authed_id}."}

    if req_id != authed_id:
        return {"error": "FORBIDDEN", "message": f"Cross-customer access denied. Use authenticated customer_id={authed_id}."}

    return None 


# ---------------- Tools ----------------
@tool
def get_customer_info(customer_id: int):
    """auth_required: look up a customer row by CustomerID (requires auth)."""
    scope_error = enforce_customer_scope(customer_id)
    if scope_error:
        return scope_error
    return db.run(f"SELECT * FROM Customer WHERE CustomerID = {customer_id};")

@tool
def get_past_purchases(
    customer_id: int,
    limit: int = 25,
    recent_first: bool = True,
):
    """
    Protected: fetch a customer's past purchases (invoice-level + line items) with track/album/artist/genre.
    Results are ordered by invoice date (desc by default) and limited.
    Returns: {"columns": [...], "rows": [[...], ...]}
    """
    scope_error = enforce_customer_scope(customer_id)
    if scope_error:
        return scope_error
    try:
        order = "DESC" if recent_first else "ASC"
        sql = text(f"""
            SELECT
                i.InvoiceId,
                i.InvoiceDate,
                i.Total,
                il.InvoiceLineId,
                il.UnitPrice,
                il.Quantity,
                t.TrackId,
                t.Name AS TrackName,
                g.Name AS Genre,
                a.Title AS AlbumTitle,
                ar.Name AS ArtistName
            FROM Invoice i
            JOIN InvoiceLine il ON il.InvoiceId = i.InvoiceId
            JOIN Track t ON t.TrackId = il.TrackId
            LEFT JOIN Genre g ON g.GenreId = t.GenreId
            LEFT JOIN Album a ON a.AlbumId = t.AlbumId
            LEFT JOIN Artist ar ON ar.ArtistId = a.ArtistId
            WHERE i.CustomerId = :cid
            ORDER BY i.InvoiceDate {order}, i.InvoiceId {order}, il.InvoiceLineId {order}
            LIMIT :lim
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"cid": customer_id, "lim": limit}).fetchall()

        # Normalize to {columns, rows} like db.run(..., include_columns=True)
        columns = [
            "InvoiceId", "InvoiceDate", "Total",
            "InvoiceLineId", "UnitPrice", "Quantity",
            "TrackId", "TrackName", "Genre",
            "AlbumTitle", "ArtistName",
        ]
        norm = []
        for r in rows:
            # r is a Row; index by position to keep consistent order
            norm.append([
                r[0], r[1], float(r[2]),
                r[3], float(r[4]), r[5],
                r[6], r[7], r[8],
                r[9], r[10],
            ])
        return {"columns": columns, "rows": norm}

    except Exception as e:
        return {"error": f"Query failed: {e}"}


@tool
def get_albums_by_artist(artist: str):
    """Public: return albums for a given artist."""
    return db.run(
        f"""SELECT Album.Title, Artist.Name 
            FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId 
            WHERE Artist.Name LIKE '%{artist}%';""",
        include_columns=True
    )

@tool
def get_tracks_by_artist(artist: str):
    """Public: list tracks and artist for a LIKE-matched artist."""
    return db.run(
        f"""SELECT Track.Name as SongName, Artist.Name as ArtistName 
            FROM Album 
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
            LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
            WHERE Artist.Name LIKE '%{artist}%';""",
        include_columns=True
    )

@tool
def check_for_songs(song_title: str):
    """Public: search tracks by title using a LIKE filter."""
    return db.run(
        f"""SELECT * FROM Track WHERE Name LIKE '%{song_title}%';""",
        include_columns=True
    )


@tool
def authenticate_customer(customer_id: int, email: str):
    """Authenticate: verify email matches record for given customer."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text("SELECT CustomerID, Email FROM Customer WHERE CustomerID = :cid"),
                {"cid": customer_id},
            ).fetchone()
        if not row:
            return {"success": False, "customer_id": None, "message": "No such customer_id."}
        cust_id, on_file_email = int(row[0]), (row[1] or "")
        provided = (email or "").strip().lower()
        on_file = on_file_email.strip().lower()
        if provided == on_file:
            return {"success": True, "customer_id": cust_id, "message": "Authenticated."}
        return {"success": False, "customer_id": None, "message": "Email does not match records."}
    except Exception as e:
        return {"success": False, "customer_id": None, "message": f"Auth error: {e}"}

AUTH_REQUIRED_TOOLS = {"get_customer_info", "get_past_purchases"}




# ---------------- Tools ----------------


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State, config: RunnableConfig):
        while True:
            configuration = config.get("configurable", {})
            passenger_id = configuration.get("passenger_id", None)
            state = {**state, "user_info": passenger_id}
            result = self.runnable.invoke(state)
            # If the LLM happens to return an empty response, we will re-prompt it
            # for an actual response.
            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break
        return {"messages": result}

# ---------------- Primary Assistant ----------------
# This is the main assistant that handles user queries and delegates to tools as needed.
llm = ChatOpenAI(model="gpt-4-turbo-preview")

primary_assistant_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful customer support assistant for a music store "
            " Use the provided tools to search for customer, music recommendations, and other information to assist the user's queries. "
            " When searching, be persistent. Expand your query bounds if the first search returns no results. "
            " If a search comes up empty, expand your search before giving up."
            " If you get an authorization error from using a tool, use authenticate_user tool to authenticate them first. If "
            "they are already authenticated and you get an error, inform them that they do not have access to that data. "
        ),
        ("placeholder", "{messages}"),
    ]
)

part_1_tools = [
check_for_songs,
get_tracks_by_artist,
get_albums_by_artist,
get_customer_info,
get_past_purchases,
]
part_1_assistant_runnable = primary_assistant_prompt | llm.bind_tools(part_1_tools)


def ensure_auth_node(state: State):
    """
    If the last AI message requested a protected tool, pause to collect credentials,
    call `authenticate_customer`, and return the auth result. 
    """
    history = state["messages"]

    # Find the most recent AI message (from a sub-agent) that may have asked for a tool.
    last_ai_msg = next((m for m in reversed(history) if isinstance(m, AIMessage)), None)
    if not last_ai_msg:
        return {"messages": []}

    # Identify whether the AI requested any protected tool (e.g., get_past_purchases).
    tool_calls = last_ai_msg.additional_kwargs.get("tool_calls", [])
    protected_call = next(
        (tc for tc in tool_calls if tc["function"]["name"] in AUTH_REQUIRED_TOOLS),
        None,
    )
    if not protected_call:
        return {"messages": []}

    # Tell the requesting tool it’s blocked on auth (good for traces / observability).
    auth_gate_msg = ToolMessage(
        name=protected_call["function"]["name"],
        tool_call_id=protected_call["id"],
        content=json.dumps({"error": "AUTH_REQUIRED"}),
    )

    # Just-in-time credential collection.
    customer_id = interrupt({"ask": "Enter your customer_id:", "field": "customer_id"})
    email = interrupt({"ask": "Enter your email:", "field": "email"})

    # Create an AIMessage that calls the *specific* tool we need: authenticate_customer.
    auth_call_ai = AIMessage(
        name="auth_request",
        content="Authenticating...",
        additional_kwargs={
            "tool_calls": [{
                "id": "auth",
                "type": "function",
                "function": {
                    "name": "authenticate_customer",
                    "arguments": json.dumps({"customer_id": customer_id, "email": email}),
                },
            }]
        },
    )

    # Execute only `authenticate_customer`, with fallback error handling.
    auth_runner = create_tool_node_with_fallback([authenticate_customer])
    exec_state = {**state, "messages": history + [auth_gate_msg, auth_call_ai]}
    tool_outputs = auth_runner.invoke(exec_state)["messages"]

    # Parse the tool's response to determine auth status.
    auth_success = False
    authed_customer_id: Optional[int] = None
    for msg in reversed(tool_outputs):
        if getattr(msg, "name", None) == "authenticate_customer":
            payload = msg.content
            if isinstance(payload, str):
                try:
                    payload = json.loads(payload)
                except Exception:
                    payload = {}
            auth_success = bool(payload.get("success"))
            authed_customer_id = payload.get("customer_id")
            break

    # Return all messages we emitted plus auth flags in state.
    return {
        "messages": [auth_gate_msg, auth_call_ai] + tool_outputs,
        "is_authed": auth_success,
        "customer_id": authed_customer_id,
    }

# ---------------- Routers ----------------

def _after_tools_router(state: State):
    last = state["messages"][-1]
    if isinstance(last, ToolMessage):
        payload = last.content
        if isinstance(payload, str):
            try: payload = json.loads(payload)
            except: payload = {}
        if isinstance(payload, dict) and payload.get("error") == "AUTH_REQUIRED":
            return "ensure_auth"
    return "assistant"


def create_graph():
    builder = StateGraph(State)


    # Define nodes: these do the work
    builder.add_node("assistant", Assistant(part_1_assistant_runnable))
    builder.add_node("tools", create_scoped_tool_node(part_1_tools)) #for now all of them have scope but only protected ones need
    builder.add_node("ensure_auth", ensure_auth_node)
    # Define edges: these determine how the control flow moves
    builder.add_edge(START, "assistant")
    builder.add_conditional_edges(
        "assistant",
        tools_condition,
    )

    builder.add_edge("ensure_auth", "assistant")
    # 3) Use a conditional edge FROM tools to read errors and route to ensure_auth if needed
    builder.add_conditional_edges("tools", _after_tools_router, {
        "ensure_auth": "ensure_auth",
        "assistant": "assistant",
    })


    return builder

# The checkpointer lets the graph persist its state
# this is a complete memory for the entire graph.
  
agent = create_graph().compile()



