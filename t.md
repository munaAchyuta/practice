Perfect. Below is a **clean, production-oriented LangGraph code skeleton** that directly implements the state machine we discussed.

This is **not toy code** — it’s structured the way you’d actually build it, but with implementations stubbed so you can plug in MongoDB, SMTP, Teams, and LLMs later.

I’ll give you:

1. **State definition**
2. **Node function skeletons**
3. **Routing functions**
4. **Graph construction**
5. **How to invoke (NEW vs RESUME)**
6. **Where SSE + persistence fit**

---

# 1. State Definition

LangGraph state should be explicit and typed.

```python
from typing import TypedDict, List, Dict, Optional, Literal


class Question(TypedDict):
    question_id: str
    agent: Literal["clarification"]
    reason: Literal["missing_fields", "ambiguous_details"]
    question_text: str
    email_sent: bool
    responded: bool
    responses: List[Dict]


class AgentState(TypedDict):
    # Identity
    user_id: str
    session_id: str
    thread_id: str

    # Ticket
    ticket: Dict

    # Status
    status: Literal["NEW", "HOLD", "SUCCESS"]
    resume_from_hold: bool

    # Validation
    missing_fields: List[str]
    ambiguity: Dict

    # Clarifications
    questions: List[Question]

    # Incoming user input (on resume)
    new_user_input: Optional[str]

    # Planner output
    plan: Dict

    # Chat output
    chat_response: Optional[str]
```

---

# 2. Node Implementations (Skeletons)

Each node is **single-responsibility**.

---

## 2.1 Load State

```python
def load_state(state: AgentState) -> AgentState:
    """
    Load state from MongoDB if exists.
    Decide whether this is a resume or new invocation.
    """
    db_state = load_from_mongo(
        state["session_id"],
        state["thread_id"]
    )

    if db_state:
        state.update(db_state)
        state["resume_from_hold"] = state["status"] == "HOLD"
    else:
        state["status"] = "NEW"
        state["resume_from_hold"] = False
        state["missing_fields"] = []
        state["ambiguity"] = {}
        state["questions"] = []

    return state
```

---

## 2.2 Merge User Response (Resume Only)

```python
def merge_user_response(state: AgentState) -> AgentState:
    """
    Attach user reply to the latest unanswered question.
    """
    if not state.get("new_user_input"):
        return state

    for q in reversed(state["questions"]):
        if not q["responded"]:
            q["responses"].append({
                "text": state["new_user_input"]
            })
            q["responded"] = True

            # Domain-specific merge
            merge_into_ticket(state["ticket"], q, state["new_user_input"])
            break

    return state
```

---

## 2.3 Analyze Required Fields (Deterministic)

```python
REQUIRED_FIELDS = ["title", "details", "priority", "team_lead_email"]


def analyze_required_fields(state: AgentState) -> AgentState:
    missing = [
        f for f in REQUIRED_FIELDS
        if not state["ticket"].get(f)
    ]

    state["missing_fields"] = missing

    if missing:
        state["status"] = "HOLD"

    return state
```

---

## 2.4 Analyze Details Ambiguity (Hybrid)

```python
def analyze_details_ambiguity(state: AgentState) -> AgentState:
    details = state["ticket"].get("details", "")

    ambiguous, reason = heuristic_check(details)

    if not ambiguous:
        ambiguous, reason = llm_check(details)

    state["ambiguity"] = {
        "details": {
            "is_ambiguous": ambiguous,
            "reason": reason
        }
    }

    if ambiguous:
        state["status"] = "HOLD"
    else:
        state["status"] = "SUCCESS"

    return state
```

---

## 2.5 Plan HOLD Actions

```python
def plan_hold_actions(state: AgentState) -> AgentState:
    send_email = False
    post_chat = True

    existing = any(
        q for q in state["questions"]
        if not q["responded"]
    )

    if not existing:
        send_email = True

        state["questions"].append({
            "question_id": generate_question_id(),
            "agent": "clarification",
            "reason": (
                "missing_fields"
                if state["missing_fields"]
                else "ambiguous_details"
            ),
            "question_text": build_question_text(state),
            "email_sent": False,
            "responded": False,
            "responses": []
        })

    state["plan"] = {
        "send_email": send_email,
        "post_chat": post_chat
    }

    return state
```

---

## 2.6 Execute Email

```python
def execute_email(state: AgentState) -> AgentState:
    for q in state["questions"]:
        if not q["email_sent"] and not q["responded"]:
            send_smtp_email(
                to=state["ticket"]["team_lead_email"],
                body=q["question_text"]
            )
            q["email_sent"] = True
            break

    return state
```

---

## 2.7 Post Clarification Chat (Terminal for HOLD)

```python
def post_clarification_chat(state: AgentState) -> AgentState:
    q = next(
        q for q in state["questions"]
        if not q["responded"]
    )

    state["chat_response"] = (
        f"⏸️ **Status: HOLD**\n\n"
        f"{q['question_text']}\n\n"
        f"Please reply in this thread to continue."
    )

    persist_to_mongo(state)
    return state
```

---

## 2.8 Post Success Chat (Terminal)

```python
def post_success_chat(state: AgentState) -> AgentState:
    state["chat_response"] = (
        "✅ **Validation successful**\n\n"
        "All required information is present and clear."
    )

    persist_to_mongo(state)
    return state
```

---

# 3. Routing Functions

```python
def route_entry(state: AgentState) -> str:
    if state["resume_from_hold"]:
        return "merge_user_response"
    return "analyze_required_fields"


def route_after_required(state: AgentState) -> str:
    if state["missing_fields"]:
        return "plan_hold_actions"
    return "analyze_details_ambiguity"


def route_after_ambiguity(state: AgentState) -> str:
    if state["status"] == "HOLD":
        return "plan_hold_actions"
    return "post_success_chat"


def route_after_merge(state: AgentState) -> str:
    return "analyze_required_fields"


def route_after_plan(state: AgentState) -> str:
    if state["plan"]["send_email"]:
        return "execute_email"
    return "post_clarification_chat"
```

---

# 4. Graph Construction (LangGraph)

```python
from langgraph.graph import StateGraph, END


builder = StateGraph(AgentState)

builder.add_node("load_state", load_state)
builder.add_node("merge_user_response", merge_user_response)
builder.add_node("analyze_required_fields", analyze_required_fields)
builder.add_node("analyze_details_ambiguity", analyze_details_ambiguity)
builder.add_node("plan_hold_actions", plan_hold_actions)
builder.add_node("execute_email", execute_email)
builder.add_node("post_clarification_chat", post_clarification_chat)
builder.add_node("post_success_chat", post_success_chat)

builder.set_entry_point("load_state")

builder.add_conditional_edges(
    "load_state",
    route_entry
)

builder.add_edge("merge_user_response", "analyze_required_fields")

builder.add_conditional_edges(
    "analyze_required_fields",
    route_after_required
)

builder.add_conditional_edges(
    "analyze_details_ambiguity",
    route_after_ambiguity
)

builder.add_conditional_edges(
    "plan_hold_actions",
    route_after_plan
)

builder.add_edge("execute_email", "post_clarification_chat")

builder.add_edge("post_clarification_chat", END)
builder.add_edge("post_success_chat", END)

graph = builder.compile()
```

---

# 5. Invocation Pattern (NEW vs RESUME)

```python
initial_state = {
    "user_id": user_id,
    "session_id": session_id,
    "thread_id": thread_id,
    "ticket": jira_ticket,
    "new_user_input": user_message  # None for first run
}

result = graph.invoke(initial_state)

return {
    "status": result["status"],
    "chat_response": result["chat_response"]
}
```

---

# 6. Where SSE fits

* SSE wraps `graph.invoke(...)`
* Once `post_clarification_chat` or `post_success_chat` runs:

  * `chat_response` is available
  * Graph reaches `END`
  * SSE stream closes

---

# 7. Why this skeleton is production-safe

* ✅ Explicit state machine
* ✅ Deterministic transitions
* ✅ Idempotent email sending
* ✅ Clean HOLD pause & resume
* ✅ No agent runs while waiting
* ✅ MongoDB is source of truth
* ✅ LangGraph only orchestrates

---

## Next (recommended)

If you want to keep going, I strongly suggest next:

**👉 “Add MongoDB optimistic locking + idempotency guards”**

or

**👉 “Implement the ambiguity LLM prompt + JSON schema validator”**

Say which one and I’ll build it out fully.
