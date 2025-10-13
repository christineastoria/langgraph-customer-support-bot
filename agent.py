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

# ---------------- Environment / Model ----------------
load_dotenv()
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o")

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

# ---------------- Tools ----------------
@tool
def get_customer_info(customer_id: int):
    """auth_required: look up a customer row by CustomerID (requires auth)."""
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

# ---------------- Prompts ----------------
customer_prompt = """You help users access or update account data. 
Do NOT ask for authentication directly; the system handles that automatically."""
song_prompt = """You are the Music agent.

Your goal: recommend music.

Decision policy:
- If the user explicitly asks for personalized recommendations OR agrees to them, call the protected tool `get_past_purchases` (the system will handle authentication). DO NOT ask for email or customer ID.
- If the user does NOT ask for personalization (or declines), provide generic recommendations using public tools (`get_albums_by_artist`, `get_tracks_by_artist`, `check_for_songs`), or a brief clarifying question about taste (genres/artists/moods) if needed.
- You may ask a single, quick opt-in question like: "Want personalized picks based on your past purchases?" If the user says yes, call `get_past_purchases`. If no, proceed generically.
- If you are already authenticated and the `customer_id` is known, pass that `customer_id` when calling protected tools.
- Never ask for authentication details (email, ID). The system will handle that."""

system_prompt = """You are a supervisor who routes between agents:
- For music: route 'music'
- For customer/account tasks: route 'customer'"""

def with_customer(msgs): return [SystemMessage(content=customer_prompt)] + msgs
def with_song(msgs): return [SystemMessage(content=song_prompt)] + msgs
def with_system(msgs): return [SystemMessage(content=system_prompt)] + msgs

from pydantic import BaseModel, Field
class Router(BaseModel):
    choice: str = Field(description="One of: music, customer")

customer_chain = with_customer | model.bind_tools([
    get_customer_info, authenticate_customer, get_past_purchases  # ← added
])

song_chain = with_song | model.bind_tools([
    get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_past_purchases  # ← added
])
route_chain = with_system | model.bind_tools([Router])

# ---------------- Helpers ----------------
def add_name(msg, name):
    d = msg.model_dump()
    d["name"] = name
    return AIMessage(**d)

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs

# ---------------- Tool wrappers ----------------
def _make_tools_runner(tools):
    node = ToolNode(tools)
    def run(state: State):
        msgs = state.get("messages", [])
        try:
            out = node.invoke(msgs)
            return {"messages": out}
        except Exception as e:
            tool_calls = []
            last = msgs[-1] if msgs else None
            if isinstance(last, AIMessage):
                tool_calls = last.additional_kwargs.get("tool_calls", [])
            return {"messages": [ToolMessage(content=f"Error: {e}", tool_call_id=tc["id"]) for tc in tool_calls]}
    return RunnableLambda(run)

_public_tools = _make_tools_runner([
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    authenticate_customer,
])
_auth_required_tools = _make_tools_runner([get_customer_info,  get_past_purchases])

# ---------------- Nodes ----------------
def init_node(state: State):
    return {"messages": state.get("messages", []), "is_authed": state.get("is_authed", False)}

def supervisor_node(state: State):
    ai = route_chain.invoke(state["messages"])
    return {"messages": [add_name(ai, "supervisor")]}

def router_ack_node(state: State):
    """Send ToolMessages acknowledging supervisor tool_calls before routing."""
    msgs = state.get("messages", [])
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    if not last_ai:
        return {"messages": []}
    tcs = last_ai.additional_kwargs.get("tool_calls", [])
    if not tcs:
        return {"messages": []}
    return {"messages": [
        ToolMessage(name=tc["function"]["name"], tool_call_id=tc["id"], content="ack")
        for tc in tcs
    ]}

def music_node(state: State):
    ai = song_chain.invoke(state["messages"])
    return {"messages": [add_name(ai, "music")]}

def customer_node(state: State):
    ai = customer_chain.invoke(state["messages"])
    return {"messages": [add_name(ai, "customer")]}

def public_tools_node(state: State): return _public_tools.invoke(state)
def auth_required_tools_node(state: State): return _auth_required_tools.invoke(state)

def ensure_auth_node(state: State):
    msgs = state["messages"]
    last_ai = next((m for m in reversed(msgs) if isinstance(m, AIMessage)), None)
    if not last_ai:
        return {"messages": []}
    tcs = last_ai.additional_kwargs.get("tool_calls", [])
    tc = next((t for t in tcs if t["function"]["name"] in AUTH_REQUIRED_TOOLS), None)
    if not tc:
        return {"messages": []}
    # reply first
    reply = ToolMessage(
        name=tc["function"]["name"],
        tool_call_id=tc["id"],
        content=json.dumps({"error": "AUTH_REQUIRED"}),
    )
    cid = interrupt({"ask": "Enter your customer_id:", "field": "customer_id"})
    email = interrupt({"ask": "Enter your email:", "field": "email"})
    auth_ai = AIMessage(
        content="Authenticating...",
        additional_kwargs={
            "tool_calls": [{
                "id": "auth",
                "type": "function",
                "function": {
                    "name": "authenticate_customer",
                    "arguments": json.dumps({"customer_id": cid, "email": email}),
                },
            }]
        },
        name="auth_request",
    )
    convo = msgs + [reply, auth_ai]
    auth_res = _public_tools.invoke({**state, "messages": convo})["messages"]
    ok, cust = False, None
    for m in reversed(auth_res):
        if getattr(m, "name", None) == "authenticate_customer":
            data = m.content
            if isinstance(data, str):
                try: data = json.loads(data)
                except: data = {}
            ok = data.get("success", False)
            cust = data.get("customer_id")
            break
    return {"messages": [reply, auth_ai] + auth_res, "is_authed": ok, "customer_id": cust}

# ---------------- Routing ----------------
def _route(state: State):
    msgs = state.get("messages", [])
    if not msgs: return "supervisor"
    last = msgs[-1]
    if isinstance(last, AIMessage):
        if last.name == "supervisor":
            if _is_tool_call(last): return "router_ack"
            return END
        if last.name in {"customer", "music"}:
            if _is_tool_call(last):
                tcs = last.additional_kwargs.get("tool_calls", [])
                if any(tc["function"]["name"] in AUTH_REQUIRED_TOOLS for tc in tcs):
                    return "ensure_auth" if not state.get("is_authed", False) else "auth_required_tools"
                return "public_tools"
            return "supervisor"
    if isinstance(last, ToolMessage):
        prev_ai = next((m for m in reversed(msgs[:-1]) if isinstance(m, AIMessage)), None)
        if prev_ai and prev_ai.name == "supervisor" and _is_tool_call(prev_ai):
            tcs = prev_ai.additional_kwargs.get("tool_calls", [])
            if tcs:
                try:
                    choice = json.loads(tcs[0]["function"].get("arguments")).get("choice")
                except: choice = None
                if choice in {"music", "customer"}: return choice
        return "supervisor"
    return "supervisor"

# ---------------- Graph ----------------
def create_graph():
    g = StateGraph(State)
    g.add_node("init", init_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("router_ack", router_ack_node)
    g.add_node("music", music_node)
    g.add_node("customer", customer_node)
    g.add_node("public_tools", public_tools_node)
    g.add_node("auth_required_tools", auth_required_tools_node)
    g.add_node("ensure_auth", ensure_auth_node)

    nodes = {
        "supervisor": "supervisor",
        "router_ack": "router_ack",
        "music": "music",
        "customer": "customer",
        "public_tools": "public_tools",
        "auth_required_tools": "auth_required_tools",
        "ensure_auth": "ensure_auth",
        END: END,
    }

    g.add_edge(START, "init")
    g.add_edge("init", "supervisor")
    g.add_conditional_edges("supervisor", _route, nodes)
    g.add_conditional_edges("router_ack", _route, nodes)
    g.add_conditional_edges("music", _route, nodes)
    g.add_conditional_edges("customer", _route, nodes)
    g.add_edge("public_tools", "supervisor")
    g.add_edge("auth_required_tools", "supervisor")
    g.add_edge("ensure_auth", "supervisor")
    return g

agent = create_graph().compile()
