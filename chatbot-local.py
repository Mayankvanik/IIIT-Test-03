"""
chatbot-local.py
==========
LangGraph-powered Q&A + analysis chatbot for the CUAD contract dataset.
Uses LOCAL Qdrant and LOCAL embeddings for retrieval (via retrieval-local.py),
but retains OpenAI (GPT-4o) for the LLM reasoning nodes.

Run:
  python chatbot-local.py
"""

from __future__ import annotations

import json
import os
import re
from typing import Annotated, Literal, Optional, TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

# ─── LOCAL RETRIEVAL IMPORT ───────────────────────────────
# We use importlib because the literal module name 'retrieval-local' contains a hyphen.
import importlib
retrieval_local = importlib.import_module("retrieval-local")

fetch_all_chunks = retrieval_local.fetch_all_chunks
list_contracts = retrieval_local.list_contracts
retrieve_cross_contract = retrieval_local.retrieve_cross_contract
retrieve_for_contract = retrieval_local.retrieve_for_contract
# ──────────────────────────────────────────────────────────

from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────
# Qdrant configs are natively handled inside retrieval-local.py now.
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


# ─── LLM ──────────────────────────────────────────────────

llm      = ChatOpenAI(model="gpt-4o-mini",    temperature=0,   api_key=OPENAI_API_KEY)
llm_fast = ChatOpenAI(model="gpt-4o-mini",    temperature=0,   api_key=OPENAI_API_KEY)

# Use GPT-4o for extraction/comparison tasks that need deep reasoning
llm_strong = ChatOpenAI(model="gpt-4o",       temperature=0,   api_key=OPENAI_API_KEY)

# ─── 41 CUAD Clause Categories ────────────────────────────
CUAD_CLAUSES = [
    "Governing Law", "Termination for Convenience", "Cap on Liability",
    "Non-Compete", "IP Ownership Assignment", "Revenue/Profit Sharing",
    "Minimum Commitment", "Audit Rights", "Indemnification",
    "Limitation of Liability", "Exclusivity", "Non-Solicitation",
    "Change of Control", "Renewal Term", "Notice Period to Terminate Renewal",
    "Anti-Assignment", "License Grant", "Warranty Duration",
    "Insurance", "Covenant Not to Sue", "Third Party Beneficiary",
    "Price Restrictions", "Most Favored Nation", "Competitive Restriction Exception",
    "Source Code Escrow", "Post-Termination Services", "Liquidated Damages",
    "Arbitration", "Uncapped Liability", "Affiliate License-Licensee",
    "Affiliate License-Licensor", "Irrevocable or Perpetual License",
    "Volume Restriction", "Joint IP Ownership", "Unlimited/All-You-Can-Eat License",
    "ROFR/ROFO/ROFN", "Sponsorship", "Development Agreement",
    "Parties", "Agreement Date", "Effective Date",
]

# Clause → Qdrant boolean payload key mapping
CLAUSE_FLAG_MAP = {
    "Governing Law":               "has_governing_law",
    "Termination for Convenience": "has_termination",
    "Cap on Liability":            "has_cap_liability",
    "Non-Compete":                 "has_non_compete",
    "Indemnification":             "has_indemnification",
    "Arbitration":                 "has_arbitration",
    "IP Ownership Assignment":     "has_ip_ownership",
    "Exclusivity":                 "has_exclusivity",
}

# Risk rules: clauses whose ABSENCE is a risk signal
HIGH_RISK_IF_ABSENT = [
    "Cap on Liability",
    "Termination for Convenience",
    "Governing Law",
]
MEDIUM_RISK_IF_ABSENT = [
    "Limitation of Liability",
    "Audit Rights",
    "Renewal Term",
]


# ─── Graph State ──────────────────────────────────────────
class State(TypedDict):
    # Conversation history (LangGraph reducer appends messages)
    messages: Annotated[list, add_messages]

    # Routing
    intent: Optional[str]           # "qa" | "extract" | "compare" | "risk"
    contract_name: Optional[str]    # single contract in scope
    contract_names: list[str]       # multiple contracts (compare / risk)
    clause_type: Optional[str]      # specific clause requested

    # Retrieval outputs
    retrieved_chunks: list[dict]    # flat list for QA / extract
    cross_contract_chunks: dict     # {contract_name: [chunks]} for compare/risk

    # Analysis outputs
    clause_results: dict            # {clause_type: {present, text, page, explanation}}
    risk_report: dict               # {contract_name: {score, flags, missing}}
    comparison_table: str           # formatted comparison string

    # Generation
    draft_answer: str
    grade_score: float              # 0–1; < 0.6 triggers re-retrieval
    retry_count: int
    final_answer: str


def _default_state() -> State:
    return State(
        messages=[],
        intent=None,
        contract_name=None,
        contract_names=[],
        clause_type=None,
        retrieved_chunks=[],
        cross_contract_chunks={},
        clause_results={},
        risk_report={},
        comparison_table="",
        draft_answer="",
        grade_score=1.0,
        retry_count=0,
        final_answer="",
    )


# ─── Helper: format chunks for prompt ─────────────────────
def _fmt_chunks(chunks: list[dict], max_chars: int = 6000) -> str:
    parts = []
    total = 0
    for i, c in enumerate(chunks):
        snippet = (
            f"[Chunk {i+1} | Contract: {c['contract_name']} | "
            f"Pages {c['page_start']}–{c['page_end']}]\n{c['text']}"
        )
        if total + len(snippet) > max_chars:
            break
        parts.append(snippet)
        total += len(snippet)
    return "\n\n---\n\n".join(parts)


# ═══════════════════════════════════════════════════════════
# NODE 1 — ROUTER
# ═══════════════════════════════════════════════════════════
def router_node(state: State) -> State:
    """
    Classify the latest user message into one of four intents and
    extract entity hints (contract names, clause types).
    """
    last_msg = state["messages"][-1].content

    available_contracts = list_contracts()[:30]  # sample for context

    system = """You are a router for a legal contract analysis system.
Classify the user message into exactly one intent:
  - "extract"  : user wants clause(s) extracted from a specific contract
  - "qa"       : user asks a question about a specific contract
  - "compare"  : user wants to compare clauses across multiple contracts
  - "risk"     : user wants risk analysis / flagging of a contract or set of contracts

Also extract:
  - contract_name  : the single contract name mentioned (or null)
  - contract_names : list of contract names if multiple mentioned (or [])
  - clause_type    : specific clause type mentioned (or null)

Respond ONLY with valid JSON, no markdown:
{
  "intent": "...",
  "contract_name": "...",
  "contract_names": [],
  "clause_type": "..."
}"""

    resp = llm_fast.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Available contracts (sample): {available_contracts[:10]}\n\nUser: {last_msg}"),
    ])

    try:
        parsed = json.loads(resp.content)
    except json.JSONDecodeError:
        parsed = {"intent": "qa", "contract_name": None, "contract_names": [], "clause_type": None}

    return {
        **state,
        "intent":          parsed.get("intent", "qa"),
        "contract_name":   parsed.get("contract_name"),
        "contract_names":  parsed.get("contract_names", []),
        "clause_type":     parsed.get("clause_type"),
        "retry_count":     state.get("retry_count", 0),
    }


# ═══════════════════════════════════════════════════════════
# NODE 2 — RETRIEVE
# ═══════════════════════════════════════════════════════════
def retrieve_node(state: State) -> State:
    """
    Hybrid Qdrant retrieval — behaviour depends on intent:
      qa / extract  → single-contract hybrid search
      compare/risk  → cross-contract parallel search
    """
    last_msg       = state["messages"][-1].content
    intent         = state["intent"]
    contract_name  = state["contract_name"]
    contract_names = state["contract_names"]
    clause_type    = state["clause_type"]

    # Determine the search query
    if clause_type:
        # Augment query with legal synonyms for better recall
        query = f"{clause_type} clause: {last_msg}"
    else:
        query = last_msg

    # Clause-level pre-filter (use payload boolean if available)
    flag_key = CLAUSE_FLAG_MAP.get(clause_type) if clause_type else None

    if intent in ("compare", "risk"):
        names = contract_names or (
            [contract_name] if contract_name else list_contracts()[:5]
        )
        cross = retrieve_cross_contract(query, names, top_k_per=4)
        return {**state, "cross_contract_chunks": cross, "retrieved_chunks": []}

    else:
        if not contract_name:
            # Broad search across all contracts
            chunks = retrieve_for_contract(query, contract_name="", top_k=8)
        else:
            chunks = retrieve_for_contract(
                query, contract_name, top_k=8, clause_filter=flag_key
            )
        return {**state, "retrieved_chunks": chunks, "cross_contract_chunks": {}}


# ═══════════════════════════════════════════════════════════
# NODE 3a — EXTRACT (clause extraction)
# ═══════════════════════════════════════════════════════════
def extract_node(state: State) -> State:
    """
    For each target clause type, run a targeted hybrid search then
    call the LLM to extract the clause text, page, and explanation.

    If clause_type is specified: extract just that one clause.
    Otherwise: extract all 10 priority clauses.
    """
    contract_name = state["contract_name"]
    clause_type   = state["clause_type"]
    chunks        = state["retrieved_chunks"]

    priority_clauses = [
        "Governing Law", "Termination for Convenience", "Cap on Liability",
        "Non-Compete", "IP Ownership Assignment", "Indemnification",
        "Audit Rights", "Arbitration", "Renewal Term", "Exclusivity",
    ]
    target_clauses = [clause_type] if clause_type else priority_clauses

    results: dict = {}

    for clause in target_clauses:
        # Targeted retrieval for this specific clause
        flag_key = CLAUSE_FLAG_MAP.get(clause)
        targeted = retrieve_for_contract(
            query=f"{clause} clause legal provision",
            contract_name=contract_name or "",
            top_k=5,
            clause_filter=flag_key,
        )
        if not targeted:
            targeted = chunks[:5]

        context = _fmt_chunks(targeted, max_chars=3000)

        system = f"""You are a legal contract analyst. Your task is to find the "{clause}" clause.

Respond ONLY with valid JSON:
{{
  "present": true/false,
  "extracted_text": "exact verbatim text of the clause (max 300 words) or null",
  "page": page_number_integer_or_null,
  "section": "section reference like '4.3' or null",
  "explanation": "plain English explanation of what this clause means and its implications (2-3 sentences)"
}}"""

        resp = llm_strong.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Contract: {contract_name}\n\nContext:\n{context}"),
        ])

        try:
            result = json.loads(resp.content)
        except json.JSONDecodeError:
            result = {"present": False, "extracted_text": None, "page": None, "section": None, "explanation": "Could not extract."}

        results[clause] = result

    return {**state, "clause_results": results}


# ═══════════════════════════════════════════════════════════
# NODE 3b — COMPARE + RISK
# ═══════════════════════════════════════════════════════════
def compare_risk_node(state: State) -> State:
    """
    Cross-contract comparison and risk flagging.

    1. For each contract, extract the target clause (or all priority clauses).
    2. Build a comparison table.
    3. Flag risks: missing critical clauses → high risk, unusual terms → medium risk.
    """
    cross_chunks  = state["cross_contract_chunks"]
    clause_type   = state["clause_type"]
    intent        = state["intent"]

    target_clauses = [clause_type] if clause_type else [
        "Cap on Liability", "Termination for Convenience",
        "Governing Law", "Indemnification", "Non-Compete",
    ]

    # ── Per-contract clause extraction ──
    all_results: dict[str, dict] = {}
    for contract_name, chunks in cross_chunks.items():
        context   = _fmt_chunks(chunks, max_chars=2500)
        clause_q  = ", ".join(target_clauses)
        system    = f"""You are a legal analyst. For each of these clause types:
{clause_q}

Respond ONLY with JSON mapping clause name to extraction result:
{{
  "Governing Law": {{"present": bool, "summary": "1-sentence summary or null"}},
  ...
}}"""
        resp = llm_strong.invoke([
            SystemMessage(content=system),
            HumanMessage(content=f"Contract: {contract_name}\n\nContext:\n{context}"),
        ])
        try:
            all_results[contract_name] = json.loads(resp.content)
        except json.JSONDecodeError:
            all_results[contract_name] = {}

    # ── Build comparison table ──
    lines = ["| Contract | " + " | ".join(target_clauses) + " |"]
    lines.append("|" + "---|" * (len(target_clauses) + 1))
    for contract_name, clauses in all_results.items():
        row = [contract_name[:40]]
        for clause in target_clauses:
            info = clauses.get(clause, {})
            if info.get("present"):
                row.append(info.get("summary", "✓")[:50])
            else:
                row.append("✗ absent")
        lines.append("| " + " | ".join(row) + " |")
    comparison_table = "\n".join(lines)

    # ── Risk scoring ──
    risk_report: dict = {}
    if intent == "risk":
        for contract_name, clauses in all_results.items():
            flags:   list[str] = []
            missing: list[str] = []
            score = 0

            for clause in HIGH_RISK_IF_ABSENT:
                if not clauses.get(clause, {}).get("present", False):
                    flags.append(f"🔴 HIGH: Missing '{clause}'")
                    missing.append(clause)
                    score += 3

            for clause in MEDIUM_RISK_IF_ABSENT:
                if not clauses.get(clause, {}).get("present", False):
                    flags.append(f"🟡 MEDIUM: Missing '{clause}'")
                    missing.append(clause)
                    score += 1

            risk_level = "HIGH" if score >= 3 else ("MEDIUM" if score >= 1 else "LOW")
            risk_report[contract_name] = {
                "score":      score,
                "level":      risk_level,
                "flags":      flags,
                "missing":    missing,
            }

    return {**state, "comparison_table": comparison_table, "risk_report": risk_report}


# ═══════════════════════════════════════════════════════════
# NODE 4 — ANSWER (Q&A)
# ═══════════════════════════════════════════════════════════
def answer_node(state: State) -> State:
    """
    Generate a grounded answer for Q&A intent using retrieved chunks.
    Cites exact page numbers and contract name.
    """
    last_msg      = state["messages"][-1].content
    chunks        = state["retrieved_chunks"]
    intent        = state["intent"]
    clause_results = state.get("clause_results", {})
    comparison     = state.get("comparison_table", "")
    risk_report    = state.get("risk_report", {})
    contract_name  = state["contract_name"]

    # ── Route to the right answer generator ──
    if intent == "extract" and clause_results:
        # Format clause extraction results
        parts = []
        for clause, info in clause_results.items():
            if info.get("present"):
                parts.append(
                    f"### {clause}\n"
                    f"**Present:** Yes  |  **Page:** {info.get('page', 'N/A')}  |  "
                    f"**Section:** {info.get('section', 'N/A')}\n\n"
                    f"**Extracted text:**\n> {info.get('extracted_text', '')}\n\n"
                    f"**What it means:** {info.get('explanation', '')}"
                )
            else:
                parts.append(f"### {clause}\n**Present:** Not found in the retrieved sections.")
        draft = f"## Clause Extraction Report — {contract_name}\n\n" + "\n\n---\n\n".join(parts)

    elif intent == "compare" and comparison:
        draft = f"## Cross-Contract Comparison\n\n{comparison}"

    elif intent == "risk" and risk_report:
        parts = [f"## Risk Analysis Report\n"]
        for name, info in risk_report.items():
            level = info["level"]
            emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(level, "")
            parts.append(f"### {emoji} {name}\n**Risk Level:** {level}  |  **Score:** {info['score']}")
            if info["flags"]:
                parts.append("\n".join(info["flags"]))
            if info["missing"]:
                parts.append(f"**Missing clauses:** {', '.join(info['missing'])}")
        draft = "\n\n".join(parts)

    else:
        # Standard Q&A
        context = _fmt_chunks(chunks, max_chars=5000)
        history = state["messages"][:-1][-4:]  # last 4 turns for context

        system = """You are a legal contract analyst. Answer the question based ONLY on the provided contract text.
Always cite the exact page number and contract name when referencing specific clauses.
If the answer is not found in the context, say so clearly — do not hallucinate.
Format: concise answer first, then supporting quote with citation."""

        resp = llm_strong.invoke([
            SystemMessage(content=system),
            *history,
            HumanMessage(content=f"Context:\n{context}\n\nQuestion: {last_msg}"),
        ])
        draft = resp.content

    return {**state, "draft_answer": draft}


# ═══════════════════════════════════════════════════════════
# NODE 5 — GRADE
# ═══════════════════════════════════════════════════════════
def grade_node(state: State) -> State:
    """
    Score the draft answer on two axes:
      - Groundedness  : is every claim supported by retrieved chunks?
      - Relevance     : does it actually answer the user question?

    Returns a 0–1 score. Below 0.6 → re-retrieve (up to 2 retries).
    """
    last_msg = state["messages"][-1].content
    draft    = state["draft_answer"]
    chunks   = state["retrieved_chunks"]
    intent   = state["intent"]
    retry    = state.get("retry_count", 0)

    # Skip grading for structured outputs (extract / compare / risk)
    if intent in ("extract", "compare", "risk") or not chunks:
        return {**state, "grade_score": 1.0}

    context = _fmt_chunks(chunks[:4], max_chars=2000)

    system = """Grade this answer on:
1. Groundedness (0-1): Is every factual claim supported by the context?
2. Relevance (0-1): Does the answer address the question?

Respond ONLY with JSON: {"groundedness": 0.0, "relevance": 0.0, "reason": "..."}"""

    resp = llm_fast.invoke([
        SystemMessage(content=system),
        HumanMessage(content=f"Question: {last_msg}\n\nContext:\n{context}\n\nAnswer:\n{draft}"),
    ])
    try:
        parsed = json.loads(resp.content)
        score  = (parsed["groundedness"] + parsed["relevance"]) / 2
    except Exception:
        score = 0.8  # assume OK on parse failure

    return {**state, "grade_score": score, "retry_count": retry}


# ═══════════════════════════════════════════════════════════
# NODE 6 — RESPOND
# ═══════════════════════════════════════════════════════════
def respond_node(state: State) -> State:
    """
    Finalise and optionally pretty-print the answer.
    Appends the final AIMessage to the conversation history.
    """
    draft = state["draft_answer"]
    # Light post-processing: ensure answer starts with a clear statement
    final = draft.strip()

    return {
        **state,
        "final_answer": final,
        "messages": state["messages"] + [AIMessage(content=final)],
    }


# ═══════════════════════════════════════════════════════════
# ROUTING FUNCTIONS (edges)
# ═══════════════════════════════════════════════════════════
def route_after_router(state: State) -> Literal["retrieve"]:
    return "retrieve"


def route_after_retrieve(state: State) -> Literal["extract", "compare_risk", "answer"]:
    intent = state["intent"]
    print(f"\n[Router classified intent as: {intent}]")
    if intent == "extract":
        return "extract"
    elif intent in ("compare", "risk"):
        return "compare_risk"
    else:
        return "answer"


def route_after_extract_or_compare(state: State) -> Literal["answer"]:
    return "answer"


def route_after_grade(state: State) -> Literal["retrieve", "respond"]:
    score = state.get("grade_score", 1.0)
    retry = state.get("retry_count", 0)
    if score < 0.6 and retry < 2:
        # Increment retry counter before looping back
        return "retrieve"
    return "respond"


# ═══════════════════════════════════════════════════════════
# BUILD THE GRAPH
# ═══════════════════════════════════════════════════════════
def build_graph() -> StateGraph:
    g = StateGraph(State)

    g.add_node("router",       router_node)
    g.add_node("retrieve",     retrieve_node)
    g.add_node("extract",      extract_node)
    g.add_node("compare_risk", compare_risk_node)
    g.add_node("answer",       answer_node)
    g.add_node("grade",        grade_node)
    g.add_node("respond",      respond_node)

    g.add_edge(START, "router")
    g.add_conditional_edges("router",       route_after_router)
    g.add_conditional_edges("retrieve",     route_after_retrieve)
    g.add_edge("extract",      "answer")
    g.add_edge("compare_risk", "answer")
    g.add_edge("answer",       "grade")
    g.add_conditional_edges("grade",        route_after_grade)
    g.add_edge("respond",      END)

    return g


graph = build_graph().compile()


# ═══════════════════════════════════════════════════════════
# CLI CHAT LOOP
# ═══════════════════════════════════════════════════════════
def chat():
    print("=" * 60)
    print("  CUAD Contract Analysis Chatbot (LOCAL RETRIEVAL)")
    print("  Type 'quit' to exit, 'contracts' to list available.")
    print("=" * 60)

    state = _default_state()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() == "quit":
            break
        if user_input.lower() == "contracts":
            contracts = list_contracts()
            print(f"\nAvailable contracts ({len(contracts)} total):")
            for c in contracts[:20]:
                print(f"  • {c}")
            if len(contracts) > 20:
                print(f"  ... and {len(contracts) - 20} more.")
            continue

        state["messages"] = state["messages"] + [HumanMessage(content=user_input)]

        # Run graph
        state = graph.invoke(state)

        print(f"\nAssistant:\n{state['final_answer']}")

        if state.get("grade_score", 1.0) < 0.6:
            print(f"\n[⚠️  Low confidence score: {state['grade_score']:.2f}]")


if __name__ == "__main__":
    chat()
