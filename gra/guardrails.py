"""Input guardrails: prompt-injection screening and reversible PII masking.

PII is masked *before* any text is sent to a model, and restored only in the
answer shown to the user. The model never sees the real values.
"""
import re
import unicodedata

INJECTION_PATTERNS = [
    r"(ignore|disregard|forget)\s+(all\s+|any\s+|the\s+)?(previous|prior|above|earlier)?\s*(instructions|rules)",
    r"(reveal|show|print|leak)\s+.*(system prompt|secret|password|api.?key)",
    r"you are now\s+(dan|unrestricted|jailbroken)",
    r"(disable|bypass|override)\s+.*(guardrail|safety|approval|security)",
    r"\bexfiltrat(e|ion)\b",
]

_ZERO_WIDTH = re.compile("[​-‏⁠﻿]")


def normalize(text: str) -> str:
    """Defeat trivial evasion: Unicode look-alikes, zero-width characters, spacing."""
    text = unicodedata.normalize("NFKC", text or "")
    text = _ZERO_WIDTH.sub("", text)
    return re.sub(r"\s+", " ", text).strip().lower()


def check_injection(text: str) -> list[str]:
    """Return the patterns that matched; an empty list means the input looks clean.

    A pattern screen is a first line of defence, not a guarantee. The real
    protection is that risky tools always need human approval.
    """
    clean = normalize(text)
    return [p for p in INJECTION_PATTERNS if re.search(p, clean)]


def _luhn_ok(digits: str) -> bool:
    total, parity = 0, len(digits) % 2
    for i, ch in enumerate(digits):
        d = int(ch)
        if i % 2 == parity:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


PII_PATTERNS = {
    "EMAIL": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    "CARD": re.compile(r"\b(?:\d[ -]?){13,19}\b"),
    "PHONE": re.compile(r"(?<!\w)\+?\d{1,3}[ -]?\(?\d{2,4}\)?[ -]?\d{3,4}[ -]?\d{3,4}\b"),
}


def mask_pii(text: str) -> tuple[str, dict[str, str]]:
    """Replace PII with placeholders like [EMAIL_1]. Returns (masked_text, vault)."""
    vault: dict[str, str] = {}
    counters: dict[str, int] = {}

    def replace(kind):
        def _sub(m):
            value = m.group(0)
            if kind == "CARD" and not _luhn_ok(re.sub(r"\D", "", value)):
                return value  # a long number that is not a card
            for token, original in vault.items():
                if original == value:
                    return token
            counters[kind] = counters.get(kind, 0) + 1
            token = f"[{kind}_{counters[kind]}]"
            vault[token] = value
            return token
        return _sub

    # Cards before phones, so a card number is never half-matched as a phone.
    for kind in ("EMAIL", "CARD", "PHONE"):
        text = PII_PATTERNS[kind].sub(replace(kind), text)
    return text, vault


def unmask_pii(text: str, vault: dict[str, str]) -> str:
    for token, original in vault.items():
        text = text.replace(token, original)
    return text
