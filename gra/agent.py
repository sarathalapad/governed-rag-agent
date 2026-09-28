"""The governed agent loop.

    request -> injection check -> PII masking -> route to a tool
            -> low risk: run it  |  high risk: park it for human approval
            -> answer with citations -> restore PII for the user -> audit every step
"""
import re
from pathlib import Path

from .audit import AuditLog
from .guardrails import check_injection, mask_pii, unmask_pii
from .llm import get_llm
from .retrieval import Retriever
from .tools import ApprovalQueue, build_tools

ROOT = Path(__file__).resolve().parents[1]


def rule_route(question: str) -> dict:
    """Deterministic router, used offline or when the model's choice is unusable."""
    q = question.lower()
    if re.search(r"\b(open|raise|create|log|file)\b.*\b(ticket|incident)\b", q):
        priority = "P1" if re.search(r"\b(outage|down|breach|urgent)\b", q) else "P3"
        return {"tool": "create_ticket", "args": {"summary": question, "priority": priority}}
    m = re.search(r"(?:calculate|compute|what is)\s+([\d\s.+\-*/%()]+[\d)])\s*\??$", q)
    if m and re.search(r"\d\s*[-+*/%]\s*\d", m.group(1)):
        return {"tool": "calculate", "args": {"expression": m.group(1)}}
    return {"tool": "search_policies", "args": {"query": question}}


class Agent:
    def __init__(self, docs_dir: Path = ROOT / "data" / "docs", state_dir: Path = ROOT / ".state",
                 offline: bool = False):
        self.llm, embedder = get_llm(offline)
        self.retriever = Retriever.from_folder(docs_dir, embedder=embedder,
                                               cache_path=state_dir / "embeddings.json")
        self.tools = build_tools(self.retriever, state_dir / "tickets.jsonl")
        self.approvals = ApprovalQueue(state_dir / "approvals.json")
        self.audit = AuditLog(state_dir / "audit.jsonl")

    def _choose(self, question: str) -> dict:
        choice = self.llm.route(question, self.tools.values())
        if not choice or choice.get("tool") not in self.tools or not isinstance(choice.get("args"), dict):
            choice = rule_route(question)
        return choice

    def run(self, question: str, user: str) -> dict:
        hits = check_injection(question)
        if hits:
            self.audit.append("request_blocked", user=user, reason="prompt_injection", patterns=hits)
            return {"status": "blocked", "answer": "Request blocked by the prompt-injection guardrail."}

        masked, vault = mask_pii(question)
        self.audit.append("request", user=user, question=masked, pii_masked=sorted(vault))

        choice = self._choose(masked)
        tool = self.tools[choice["tool"]]
        self.audit.append("tool_selected", user=user, tool=tool.name, risk=tool.risk, args=choice["args"])

        if tool.risk == "high":
            approval_id = self.approvals.request(tool.name, choice["args"], user)
            self.audit.append("approval_requested", user=user, approval_id=approval_id, tool=tool.name)
            return {"status": "pending_approval", "approval_id": approval_id,
                    "answer": f"'{tool.name}' is high-risk. Waiting for approval {approval_id} by another user."}

        try:
            result = tool.fn(**choice["args"])
        except Exception as e:
            self.audit.append("tool_error", user=user, tool=tool.name, error=type(e).__name__)
            return {"status": "error", "answer": f"The tool failed: {e}"}

        if tool.name == "calculate":
            answer, citations = f"{choice['args']['expression'].strip()} = {result['result']:g}", []
        else:
            docs = result["hits"]
            answer = self.llm.answer(masked, docs) if docs else "I don't know. No relevant documents found."
            citations = [f"[{i}] {h.chunk.source} - {h.chunk.section}" for i, h in enumerate(docs, 1)]

        grounded = tool.name != "search_policies" or bool(re.search(r"\[\d+\]", answer))
        self.audit.append("answer", user=user, tool=tool.name, model=self.llm.last_model,
                          grounded=grounded, citations=citations)
        return {"status": "answered", "answer": unmask_pii(answer, vault), "citations": citations,
                "grounded": grounded, "model": self.llm.last_model}

    def decide(self, approval_id: str, approver: str, approve: bool = True) -> dict:
        try:
            item = self.approvals.decide(approval_id, approver, approve)
        except (ValueError, PermissionError) as e:
            self.audit.append("approval_refused", approver=approver, approval_id=approval_id, reason=str(e))
            raise
        self.audit.append("approval_decided", approver=approver, approval_id=approval_id, status=item["status"])
        if item["status"] != "approved":
            return {"status": "rejected"}
        result = self.tools[item["tool"]].fn(**item["args"])
        self.audit.append("tool_executed", approval_id=approval_id, tool=item["tool"], result=result)
        return {"status": "executed", "result": result}
