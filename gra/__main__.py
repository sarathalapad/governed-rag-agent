"""Command line: python -m gra <command>"""
import argparse
import json
import sys

from .agent import Agent


def main(argv=None):
    p = argparse.ArgumentParser(prog="gra", description="Governed RAG Agent")
    p.add_argument("--offline", action="store_true", help="no model; extractive answers")
    sub = p.add_subparsers(dest="cmd", required=True)
    ask = sub.add_parser("ask", help="ask a question or request an action")
    ask.add_argument("question")
    ask.add_argument("--user", default="alice")
    sub.add_parser("pending", help="list actions waiting for approval")
    for name in ("approve", "reject"):
        d = sub.add_parser(name, help=f"{name} a pending action")
        d.add_argument("approval_id")
        d.add_argument("--user", default="bob")
    sub.add_parser("verify-audit", help="check the audit chain for tampering")
    sub.add_parser("eval", help="run the evaluation suite")
    args = p.parse_args(argv)

    if args.cmd == "eval":
        from .evals import run_evals
        return run_evals(offline=args.offline)

    agent = Agent(offline=args.offline)
    if args.cmd == "ask":
        r = agent.run(args.question, args.user)
        print(f"[{r['status']}] {r['answer']}")
        for c in r.get("citations", []):
            print("   ", c)
        if "model" in r:
            print(f"    model: {r['model']} | grounded: {r['grounded']}")
    elif args.cmd == "pending":
        print(json.dumps(agent.approvals.pending(), indent=2) or "{}")
    elif args.cmd in ("approve", "reject"):
        try:
            r = agent.decide(args.approval_id, args.user, approve=args.cmd == "approve")
        except (ValueError, PermissionError) as e:
            print(f"refused: {e}")
            return 1
        print(json.dumps(r, indent=2, default=str))
    elif args.cmd == "verify-audit":
        ok, line = agent.audit.verify()
        print("audit chain intact" if ok else f"TAMPERING DETECTED at line {line}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
