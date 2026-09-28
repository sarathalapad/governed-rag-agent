"""Agent tools with risk levels, and a maker-checker approval queue for risky ones."""
import ast
import json
import operator
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable


@dataclass
class Tool:
    name: str
    risk: str  # "low" runs immediately; "high" needs approval by a different person
    description: str
    fn: Callable[..., dict]


_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.USub: operator.neg, ast.Mod: operator.mod}


def safe_eval(expr: str) -> float:
    """Arithmetic only. No names, calls or attributes, so no code execution."""
    def ev(node):
        if isinstance(node, ast.Expression):
            return ev(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
            if isinstance(node.op, ast.Pow) and abs(ev(node.right)) > 100:
                raise ValueError("exponent too large")
            return _OPS[type(node.op)](ev(node.left), ev(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
            return _OPS[type(node.op)](ev(node.operand))
        raise ValueError("only arithmetic is allowed")
    return ev(ast.parse(expr, mode="eval"))


def build_tools(retriever, tickets_path: Path) -> dict[str, Tool]:
    def search_policies(query: str) -> dict:
        return {"hits": retriever.search(query, k=3)}

    def calculate(expression: str) -> dict:
        return {"result": safe_eval(expression)}

    def create_ticket(summary: str, priority: str = "P3") -> dict:
        ticket = {"id": f"INC-{uuid.uuid4().hex[:6].upper()}", "summary": summary,
                  "priority": priority, "created": time.strftime("%Y-%m-%dT%H:%M:%S")}
        tickets_path.parent.mkdir(parents=True, exist_ok=True)
        with tickets_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ticket) + "\n")
        return {"ticket": ticket}

    tools = [
        Tool("search_policies", "low", "Search company policy documents. args: {query}", search_policies),
        Tool("calculate", "low", "Evaluate an arithmetic expression. args: {expression}", calculate),
        Tool("create_ticket", "high", "Open an IT incident ticket. args: {summary, priority}", create_ticket),
    ]
    return {t.name: t for t in tools}


class ApprovalQueue:
    """Pending high-risk actions. The approver must differ from the requester."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def _load(self) -> dict:
        return json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}

    def _save(self, data: dict):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def request(self, tool: str, args: dict, requester: str) -> str:
        data = self._load()
        approval_id = f"APR-{uuid.uuid4().hex[:6].upper()}"
        data[approval_id] = {"tool": tool, "args": args, "requester": requester,
                             "status": "pending", "requested": time.strftime("%Y-%m-%dT%H:%M:%S")}
        self._save(data)
        return approval_id

    def pending(self) -> dict:
        return {k: v for k, v in self._load().items() if v["status"] == "pending"}

    def decide(self, approval_id: str, approver: str, approve: bool) -> dict:
        data = self._load()
        item = data.get(approval_id)
        if not item or item["status"] != "pending":
            raise ValueError("unknown or already decided approval")
        if approver == item["requester"]:
            raise PermissionError("maker-checker: you cannot approve your own request")
        item.update(status="approved" if approve else "rejected", approver=approver,
                    decided=time.strftime("%Y-%m-%dT%H:%M:%S"))
        self._save(data)
        return item
