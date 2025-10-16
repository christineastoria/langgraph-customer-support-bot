from typing import Optional, TypedDict, Annotated, Union, List
import json, sqlite3, requests
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
from sqlalchemy.pool import StaticPool

from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
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
from langchain_core.messages import HumanMessage, SystemMessage, BaseMessage
from datetime import datetime
from typing import Optional, List, Dict, Any

# ---------------- Environment / Model ----------------
load_dotenv()
model = ChatOpenAI(temperature=0, streaming=True, model="gpt-4o").with_config({
    "run_name": "llm",
    "tags": ["llm"],
    "metadata": {"model": "gpt-4o"}
})
# model = ChatAnthropic(
#     temperature=0,
#     streaming=True,
#     model="claude-sonnet-4-5-20250929"   # e.g., Anthropic model id
# ).with_config({
#     "run_name": "llm",
#     "tags": ["llm"],
#     "metadata": {"provider": "anthropic", "model": "claude-sonnet-4-5-20250929"}
# })
graph_state_ctx: ContextVar[dict] = ContextVar("graph_state_ctx", default={})

# ---------------- State ----------------
class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    is_authed: bool
    customer_id: Optional[int]
    pending_routes: List[str] 
    resume_agent: Optional[str]   

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

def add_name(msg, name):
    d = msg.model_dump()
    d["name"] = name
    return AIMessage(**d)

def _is_tool_call(msg):
    return hasattr(msg, "additional_kwargs") and "tool_calls" in msg.additional_kwargs

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

def last_user_text(msgs: list[BaseMessage]) -> str:
    for m in reversed(msgs):
        if isinstance(m, HumanMessage):
            return m.content if isinstance(m.content, str) else str(m.content)
    return ""

def focus_system_for(agent: str, state: State) -> SystemMessage:
    latest = last_user_text(state["messages"])
    remaining = list(state.get("pending_routes", []))  # after supervisor popped the current one
    is_final = len(remaining) == 0

    return SystemMessage(content=f"""
        You are the **{agent}** agent in a multi-agent system. You can see the entire history for context.

        Respond to the **latest user message** below from the {agent} domain.
        Intents remaining after your reply: {remaining if remaining else "[]"}
        {"You are the **final** responder for this user turn." if is_final else "You are **not** the final responder for this turn; another agent will reply next."}

        CRITICAL RULES:
        - Keep responses short to medium length and to the point.
        - If tools are needed, call them; otherwise answer directly.
        - You may call at most one tool per response. Choose the single most appropriate tool; do not call multiple at once
        - Do not call the same tool more than once in a row with identical arguments; if you have the data, answer from it.
        - Provide a concrete answer to the latest user message for your domain.
        - Treat assistant messages as notes from other agents; do not assume the task is done unless **you** answered it.
        - {("Since you are final, wrap cleanly with the needed information only.") if is_final else ("Since you are not final, avoid wrap-up/closing language; keep it concise so the next agent can continue.")}

        Latest user message to address now:
        ---
        {latest}
    ---
    """)


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
    recent_first: bool = True,
):
    """
    Protected: fetch a customer's past purchases (invoice-level + line items) with track/album/artist/genre.
    Results are ordered by invoice date (desc by default).
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
        """)
        with engine.connect() as conn:
            rows = conn.execute(sql, {"cid": customer_id}).fetchall()

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
    result = db.run(
        f"""SELECT Album.Title, Artist.Name 
            FROM Album JOIN Artist ON Album.ArtistId = Artist.ArtistId 
            WHERE Artist.Name LIKE '%{artist}%';""",
        include_columns=True
    )
    if result == "":
        return {"error": "NO_RESULTS", "message": f"No artists found in the inventory matching '{artist}'."}
    return result


@tool
def get_tracks_by_artist(artist: str):
    """Public: list tracks and artist for a LIKE-matched artist."""
    result = db.run(
        f"""SELECT Track.Name as SongName, Artist.Name as ArtistName 
            FROM Album 
            LEFT JOIN Artist ON Album.ArtistId = Artist.ArtistId 
            LEFT JOIN Track ON Track.AlbumId = Album.AlbumId 
            WHERE Artist.Name LIKE '%{artist}%';""",
        include_columns=True
    )
    if result == "":
        return {"error": "NO_RESULTS", "message": f"No artists found in the inventory matching '{artist}'."}
    return result


@tool
def check_for_songs(song_title: str):
    """Public: search tracks by title using a LIKE filter."""
    result = db.run(
        f"""SELECT * FROM Track WHERE Name LIKE '%{song_title}%';""",
        include_columns=True
    )
    if result == "":
        return {"error": "NO_RESULTS", "message": f"No songs found in the inventory matching '{song_title}'."}
    return result


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
    
def get_state():
    return graph_state_ctx.get({})

def resolve_track_ids(conn, kind: str, item: str) -> List[int]:
    """
    Resolve user intent to TrackId list.
    kind: "track" | "song" | "album"
    item: can be id or name/title
    """
    kind = (kind or "track").lower()
    # numeric id shortcut
    def _is_int(s):
        try: int(s); return True
        except: return False

    if kind in {"track", "song"}:
        if _is_int(item):
            # validate track id exists
            row = conn.execute(text("SELECT TrackId FROM Track WHERE TrackId=:tid"), {"tid": int(item)}).fetchone()
            return [int(row[0])] if row else []
        # name lookup
        rows = conn.execute(text("""
            SELECT TrackId FROM Track WHERE lower(Name) LIKE :q ORDER BY TrackId
        """), {"q": f"%{item.lower()}%"}).fetchall()
        return [int(r[0]) for r in rows]

    if kind == "album":
        # id or title
        if _is_int(item):
            aid = int(item)
        else:
            row = conn.execute(text("""
                SELECT AlbumId FROM Album WHERE lower(Title) LIKE :q ORDER BY AlbumId LIMIT 1
            """), {"q": f"%{item.lower()}%"}).fetchone()
            aid = int(row[0]) if row else None
        if not aid:
            return []
        rows = conn.execute(text("SELECT TrackId FROM Track WHERE AlbumId=:aid ORDER BY TrackId"), {"aid": aid}).fetchall()
        return [int(r[0]) for r in rows]

    return []


@tool
def purchase_item(kind: str, item: str, quantity: int = 1):
    """
    Protected/Write: Create a NEW Invoice for the authed customer and add InvoiceLine(s).
    Arguments:
      - kind: "track" | "song" | "album"
      - item: track id or name; OR album id or title
      - quantity (optional): per-track quantity (default 1). For albums, applies to every track.
    Returns: {invoice_id, total, lines:[{track_id, track_name, unit_price, quantity, line_total}], invoice_date, billing:{...}}
    """
    st = get_state()
    customer_id = st.get("customer_id")
    scope_error = enforce_customer_scope(customer_id)
    if scope_error:
        return scope_error

    try:
        q = int(quantity)
        if q <= 0:
            return {"error":"INVALID_INPUT", "message":"quantity must be > 0"}
    except Exception:
        return {"error":"INVALID_INPUT", "message":"quantity must be an integer"}

    now_iso = datetime.utcnow().isoformat(timespec="seconds")
    with engine.begin() as conn:
        # resolve tracks
        track_ids = resolve_track_ids(conn, kind, item)
        if not track_ids:
            return {"error":"NOT_FOUND", "message": f"No tracks resolved for kind='{kind}' item='{item}'."}

        # customer billing for invoice
        cust = conn.execute(text("""
            SELECT FirstName, LastName, Address, City, State, Country, PostalCode
            FROM Customer WHERE CustomerId=:cid
        """), {"cid": customer_id}).fetchone()
        if not cust:
            return {"error":"NO_CUSTOMER", "message":"Customer not found."}

        # fetch prices & names
        rows = conn.execute(text(f"""
            SELECT TrackId, UnitPrice, Name FROM Track
            WHERE TrackId IN ({",".join([str(t) for t in track_ids])})
        """)).fetchall()
        price = {int(r[0]): float(r[1]) for r in rows}
        name  = {int(r[0]): r[2] for r in rows}
        missing = [t for t in track_ids if t not in price]
        if missing:
            return {"error":"PRICE_MISSING", "message": f"Missing tracks: {missing}"}

        # compute total + insert invoice
        lines: List[Dict[str, Any]] = []
        total = 0.0
        for tid in track_ids:
            unit = price[tid]; line_total = unit * q
            total += line_total
            lines.append({
                "track_id": tid,
                "track_name": name[tid],
                "unit_price": unit,
                "quantity": q,
                "line_total": round(line_total, 2),
            })

        inv = conn.execute(text("""
            INSERT INTO Invoice (
                CustomerId, InvoiceDate,
                BillingAddress, BillingCity, BillingState, BillingCountry, BillingPostalCode,
                Total
            )
            VALUES (:cid, :dt, :addr, :city, :state, :country, :postal, :total)
        """), {
            "cid": customer_id, "dt": now_iso,
            "addr": cust[2], "city": cust[3], "state": cust[4], "country": cust[5], "postal": cust[6],
            "total": round(total, 2)
        })
        invoice_id = int(inv.lastrowid)

        # insert lines
        for L in lines:
            conn.execute(text("""
                INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity)
                VALUES (:iid, :tid, :price, :qty)
            """), {"iid": invoice_id, "tid": L["track_id"], "price": L["unit_price"], "qty": L["quantity"]})

    return {
        "invoice_id": invoice_id,
        "total": round(total, 2),
        "lines": lines,
        "invoice_date": now_iso,
        "billing": {"address": cust[2], "city": cust[3], "state": cust[4], "country": cust[5], "postal": cust[6]},
    }


@tool
def refund_invoice(invoice_id: int, mode: str = "full", lines: Optional[List[Dict[str, int]]] = None):
    """
    Protected/Write: Append negative-quantity InvoiceLine(s) to an existing Invoice.
    Arguments:
      - invoice_id: the Invoice to refund (must belong to authed customer)
      - mode: "full" (default) or "partial"
      - lines (required if partial): list of {"invoice_line_id": int, "qty": int} to refund
    Behavior:
      - FULL: adds a negative line for each original line (same UnitPrice, Quantity * -1) and reduces Invoice.Total.
      - PARTIAL: for each provided line, adds a negative line with that qty; reduces Invoice.Total accordingly.
    Returns: {invoice_id, amount, mode, added_lines:[{orig_line_id, track_id, unit_price, quantity, refunded_amount}], new_invoice_total}
    """
    st = get_state()
    customer_id = st.get("customer_id")
    scope_error = enforce_customer_scope(customer_id)
    if scope_error:
        return scope_error

    mode = (mode or "full").lower()
    if mode not in {"full", "partial"}:
        return {"error":"INVALID_INPUT", "message":"mode must be 'full' or 'partial'."}
    if mode == "partial":
        if not isinstance(lines, list) or not lines:
            return {"error":"INVALID_INPUT", "message":"Provide 'lines' for partial refunds."}

    with engine.begin() as conn:
        inv = conn.execute(text("""
            SELECT InvoiceId, CustomerId, Total FROM Invoice WHERE InvoiceId=:iid
        """), {"iid": invoice_id}).fetchone()
        if not inv or int(inv[1]) != int(customer_id):
            return {"error":"FORBIDDEN", "message":"Invoice not found for this customer."}

        # map original lines
        orig = conn.execute(text("""
            SELECT InvoiceLineId, TrackId, UnitPrice, Quantity
            FROM InvoiceLine WHERE InvoiceId=:iid
        """), {"iid": invoice_id}).fetchall()
        orig_map = {int(r[0]): {"track_id": int(r[1]), "unit_price": float(r[2]), "quantity": int(r[3])} for r in orig}
        if mode == "full" and not orig_map:
            return {"error":"NO_LINES", "message":"Invoice has no lines to refund."}

        added = []
        refund_amount = 0.0

        if mode == "full":
            to_refund = [{ "invoice_line_id": lid, "qty": data["quantity"] } for lid, data in orig_map.items()]
        else:
            to_refund = lines

        for item in to_refund:
            lid = int(item["invoice_line_id"])
            qty = int(item["qty"])
            if lid not in orig_map:
                return {"error":"INVALID_LINE", "message": f"InvoiceLineId {lid} not in invoice."}
            if qty <= 0 or qty > orig_map[lid]["quantity"]:
                return {"error":"INVALID_QTY", "message": f"qty {qty} invalid for line {lid} (max {orig_map[lid]['quantity']})."}

            track_id   = orig_map[lid]["track_id"]
            unit_price = orig_map[lid]["unit_price"]
            # negative quantity line
            conn.execute(text("""
                INSERT INTO InvoiceLine (InvoiceId, TrackId, UnitPrice, Quantity)
                VALUES (:iid, :tid, :price, :negqty)
            """), {"iid": invoice_id, "tid": track_id, "price": unit_price, "negqty": -qty})

            refunded_amount = unit_price * qty
            refund_amount += refunded_amount
            added.append({
                "orig_line_id": lid,
                "track_id": track_id,
                "unit_price": unit_price,
                "quantity": -qty,
                "refunded_amount": round(refunded_amount, 2),
            })

        # reduce invoice total
        new_total = round(float(inv[2]) - refund_amount, 2)
        conn.execute(text("UPDATE Invoice SET Total=:tot WHERE InvoiceId=:iid"),
                     {"tot": new_total, "iid": invoice_id})

    return {
        "invoice_id": int(invoice_id),
        "amount": round(refund_amount, 2),
        "mode": mode,
        "added_lines": added,
        "new_invoice_total": new_total,
    }

AUTH_REQUIRED_TOOLS = {"get_customer_info", "get_past_purchases", "purchase_item", "refund_invoice"}
WRITE_ACCESS_TOOLS = {"purchase_item", "refund_invoice"}

# ---------------- Prompts ----------------
customer_prompt = """You help users access or update account data. 
Do NOT ask for authentication directly; the system handles that automatically.
- Use get_customer_info to fetch customer personal details like address, email, name, etc. 
- You can treat any assistant messages as notes and help from other agents; do **not** assume the request is already resolved unless **you** have responded yourself.
- Do not execute a purchase or refund unless you the user has specified explicitly what item they want to purchase or refund. If they haven't, ask. 
- Use purchase_item or refund_invoice when the user explicitly requests a purchase or refund. Don't try to execute these actions yourself
"""

song_prompt = """You are the Music agent.

Your goal: recommend music for purchase.
- Offer personalized reccommendations when relevant. If the user explicitly asks for personalized recommendations OR agrees to them, call the protected tool `get_past_purchases` (the system will handle authentication). DO NOT ask for email or customer ID.
- If the user does NOT ask for personalization (or declines), provide recommendations for music we have at this store by using public tools (`get_albums_by_artist`, `get_tracks_by_artist`, `check_for_songs`), or a brief clarifying question about taste (genres/artists/moods) if needed.
- Prioritize recommendations that are in the store's catalog from the public tools when possible, but dont continue to search for artists that you find out aren't there.
- If public tools return None, then those queried items aren't available in the store. 
- If recommending, always ask if the user wants to purchase recommendations
- You may ask a single, quick opt-in question like: "Want personalized picks based on your past purchases?" If the user says yes, call `get_past_purchases`. If no, proceed generically.
- If you are already authenticated and the `customer_id` is known, pass that `customer_id` when calling protected tools.
- Never ask for authentication details (email, ID). The system will handle that.
- Answer the **latest user message below** from the perspective of the music domain.
- You can treat any assistant messages as notes and help from other agents; do **not** assume the request is already resolved unless **you** have responded yourself.
"""

#TODO: make this a prompt template with outputs required from a pydantic model
system_prompt = """You are a supervisor of a music store who routes between agents:
- For music or any personalized music recommendations: route ['music']
- For other customer/account tasks: route ['customer']
- Prioritize single routes, but if more than one applies, you can route both. Example: ['customer', 'music']. If more than one applies, put them in order of desired execution. 
"""


def with_system(msgs): return [SystemMessage(content=system_prompt)] + msgs

from pydantic import BaseModel, Field

class Router(BaseModel):
    step: Union[Literal["music","customer"], List[Literal["music","customer"]]] = Field(
        description="One or more next steps."
    )

router = model.with_structured_output(Router).with_config(
    {
    "run_name": "router",
    "tags": ["router", "supervisor"],
    "metadata": {"component": "router", "agent": "supervisor"}
    }
)

def with_customer_fn(msgs): 
    return [SystemMessage(content=customer_prompt)] + msgs

def with_song_fn(msgs): 
    return [SystemMessage(content=song_prompt)] + msgs

customer_chain = RunnableLambda(with_customer_fn) | model.bind_tools([
    get_customer_info, authenticate_customer, get_past_purchases, purchase_item, refund_invoice, 
]).with_config({
    "run_name": "agent:customer.llm",
    "tags": ["agent:customer", "llm"],
    "metadata": {"component": "agent_llm", "agent": "customer"}
})

song_chain = RunnableLambda(with_song_fn) | model.bind_tools([
    get_albums_by_artist, get_tracks_by_artist, check_for_songs, get_past_purchases
]).with_config({
    "run_name": "agent:music.llm",
    "tags": ["agent:music", "llm"],
    "metadata": {"component": "agent_llm", "agent": "music"}
})


# ---------------- Tool wrappers ----------------

public_tools = [
    get_albums_by_artist,
    get_tracks_by_artist,
    check_for_songs,
    authenticate_customer,
]
auth_required_tools = [get_customer_info,  get_past_purchases]
write_access_tools = [purchase_item, refund_invoice]

def handle_tool_error(state) -> dict:
    error = state.get("error")
    last = state.get("messages", [])[-1] if state.get("messages") else None
    tool_calls = getattr(last, "tool_calls", []) or getattr(last, "additional_kwargs", {}).get("tool_calls", []) or []
    if not tool_calls:
        # No specific tool call to respond to—emit a generic ToolMessage for tracing
        return {"messages": [ToolMessage(content=f"Error: {repr(error)}", tool_call_id="unknown")]}
    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease fix your mistakes.",
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


# ---------------- Nodes ----------------
def init_node(state: State):
    return {
        "messages": state.get("messages", []),
        "is_authed": state.get("is_authed", False),
        "customer_id": state.get("customer_id"),
        "pending_routes": [],
        "resume_agent": None,            
    }

def supervisor_node(state: State):
    # If there’s already a queue, don’t recompute; just keep it.
    if state.get("pending_routes"):
        return {"pending_routes": state["pending_routes"]}

    decision = router.invoke([SystemMessage(content=system_prompt)] + state["messages"])
    steps = decision.step if isinstance(decision.step, list) else [decision.step]
    return {
        "pending_routes": steps,
        # optional visibility only:
        # "__metadata__": {"supervisor_decision": steps},
        # "messages": [AIMessage(content=f"(Supervisor decision: {steps})", name="supervisor")],
    }

def music_node(state: State):
    msgs = state["messages"] + [focus_system_for("music", state)]
    ai = song_chain.invoke(msgs)
    return {"messages": [add_name(ai, "music")]}

def customer_node(state: State):
    msgs =  state["messages"] + [focus_system_for("customer", state)]
    ai = customer_chain.invoke(msgs)
    return {"messages": [add_name(ai, "customer")]}


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

    requester = getattr(last_ai_msg, "name", None)

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
        "pending_routes": [],
        "resume_agent": requester if auth_success and requester in {"music", "customer"} else None,
    }


# ---------------- Routing ----------------
def route_from_supervisor(state: State):
    # 1) resume after auth still wins
    resume = state.get("resume_agent")
    if resume in {"music","customer"}:
        state["resume_agent"] = None
        return resume

    # 2) pop next pending route if any
    pending = state.get("pending_routes", [])
    if isinstance(pending, list) and pending:
        nxt = pending.pop(0)
        state["pending_routes"] = pending
        return nxt

    # 3) nothing pending → idle here (lets graph END naturally if no more work)
    return "supervisor"

def _tool_names(msg):
    tcs = msg.additional_kwargs.get("tool_calls", [])
    return [tc["function"]["name"] for tc in tcs]

def _has_pending(state: State) -> bool:
    prs = state.get("pending_routes", [])
    return isinstance(prs, list) and len(prs) > 0

def route_from_music(state: State):
    last = state["messages"][-1]
    if _is_tool_call(last):
        names = _tool_names(last)

        # Gate auth first if *any* requested tool needs it
        if any(n in AUTH_REQUIRED_TOOLS for n in names) and not state.get("is_authed", False):
            return "ensure_auth"
        
        # Otherwise protected read-only tools
        if any(n in AUTH_REQUIRED_TOOLS for n in names):
            return "auth_required_tools"

        # Otherwise public
        return "public_tools"

    return "supervisor" if _has_pending(state) else END

def route_from_customer(state: State):
    last = state["messages"][-1]
    if _is_tool_call(last):
        names = _tool_names(last)

        # Gate auth first if *any* requested tool needs it
        if any(n in AUTH_REQUIRED_TOOLS for n in names) and not state.get("is_authed", False):
            return "ensure_auth"

        # Then prefer write node for writes (now authed)
        if any(n in WRITE_ACCESS_TOOLS for n in names):
            return "write_access_tools"

        # Otherwise protected read-only tools
        if any(n in AUTH_REQUIRED_TOOLS for n in names):
            return "auth_required_tools"

        # Otherwise public
        return "public_tools"

    return "supervisor" if _has_pending(state) else END

def route_from_tools(state: State):
    """After any ToolMessage, return to the agent that asked for it, else supervisor."""
    msgs = state.get("messages", [])
    prev_ai = next((m for m in reversed(msgs[:-1])
                    if isinstance(m, AIMessage) and _is_tool_call(m)), None)
    if prev_ai and prev_ai.name in {"music","customer"}:
        return prev_ai.name
    return "supervisor"




# ---------------- Graph ----------------
def create_graph():
    g = StateGraph(State)
    g.add_node("init", init_node)
    g.add_node("supervisor", supervisor_node)
    g.add_node("music", music_node)
    g.add_node("customer", customer_node)
    g.add_node("public_tools", create_tool_node_with_fallback(public_tools))
    g.add_node("auth_required_tools", create_scoped_tool_node(auth_required_tools))
    g.add_node("write_access_tools", create_scoped_tool_node(write_access_tools))
    g.add_node("ensure_auth", ensure_auth_node)

    g.add_edge(START, "init")
    g.add_edge("init", "supervisor")
    g.add_conditional_edges("supervisor", route_from_supervisor, {
    "supervisor":"supervisor", "music":"music", "customer":"customer"
    })
    g.add_conditional_edges("music", route_from_music, {
        END: END, "public_tools":"public_tools",
        "ensure_auth":"ensure_auth", "auth_required_tools":"auth_required_tools", "write_access_tools":"write_access_tools", "supervisor": "supervisor"
    })
    g.add_conditional_edges("customer", route_from_customer, {
        END: END, "public_tools":"public_tools",
        "ensure_auth":"ensure_auth", "auth_required_tools":"auth_required_tools", "write_access_tools":"write_access_tools", "supervisor": "supervisor"
    })
    # After any tools finish, decide where to go:
    g.add_conditional_edges("public_tools", route_from_tools,
        {"music":"music","customer":"customer"})
    g.add_conditional_edges("auth_required_tools", route_from_tools,
        {"music":"music","customer":"customer"})
    g.add_conditional_edges("write_access_tools", route_from_tools,
        {"music":"music","customer":"customer"})
    g.add_conditional_edges("ensure_auth", route_from_supervisor,  
        {"supervisor":"supervisor","music":"music","customer":"customer"})
    return g

agent = create_graph().compile(interrupt_before=["write_access_tools"])
