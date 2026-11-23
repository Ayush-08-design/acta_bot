import re
import unicodedata


def _is_valid_luhn(card_number: str) -> bool:
    """Validates a credit card number using the Luhn checksum algorithm."""
    digits = [int(d) for d in re.sub(r"\D", "", card_number)]
    if not (13 <= len(digits) <= 19):
        return False

    checksum = 0
    for i, digit in enumerate(reversed(digits)):
        if i % 2 == 1:
            digit *= 2
            if digit > 9:
                digit -= 9
        checksum += digit
    return checksum % 10 == 0


# Pre-compiled patterns for speed
_EMAIL_RE = re.compile(
    r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", re.IGNORECASE
)

# Excludes invalid SSN prefixes (000, 666, 900-999) and zero group/serial numbers
_SSN_RE = re.compile(r"\b(?!000|666|9\d{2})\d{3}[- ]?(?!00)\d{2}[- ]?(?!0000)\d{4}\b")

# Matches 13-19 digits separated by single spaces or hyphens (verified via Luhn)
_CARD_CANDIDATE_RE = re.compile(r"\b(?:\d[ -]?){12,18}\d\b")

# Common phone formats (US/International)
_PHONE_RE = re.compile(
    r"(?<!\w)(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]\d{3}[-.\s]\d{4}\b"
)


def redact_pii(text: str) -> str:
    """Safely redacts Emails, SSNs, Luhn-verified Credit Cards, and Phone numbers."""
    if not text:
        return text

    # 1. Credit Cards (Check first before SSN/Phone patterns eat partial digits)
    def _replace_card(match: re.Match) -> str:
        candidate = match.group(0)
        return "[REDACTED_CARD]" if _is_valid_luhn(candidate) else candidate

    text = _CARD_CANDIDATE_RE.sub(_replace_card, text)

    # 2. SSNs
    text = _SSN_RE.sub("[REDACTED_SSN]", text)

    # 3. Emails
    text = _EMAIL_RE.sub("[REDACTED_EMAIL]", text)

    # 4. Phone Numbers
    text = _PHONE_RE.sub("[REDACTED_PHONE]", text)

    return text


# PROMPT INJECTION DETECTION ---

_INVISIBLE_CHARS_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\ufeff\xad]")

_INJECTION_PATTERNS = [
    # Instruction overrides
    r"\b(ignore|disregard|forget|override|bypass)\b.{0,25}\b(previous|prior|above|all|system|initial)\b.{0,25}\b(instructions?|prompts?|rules?|directives?|constraints?)\b",
    # System prompt extraction / leakage
    r"\b(reveal|show|print|output|repeat|dump|display|tell me)\b.{0,25}\b(system\s*prompt|initial\s*instructions?|hidden\s*rules?|developer\s*message)\b",
    # Persona / Jailbreak switches
    r"\b(you are now|act as|pretend to be|switch to|enter)\b.{0,20}\b(dan|do anything now|developer mode|jailbreak|unrestricted|unfiltered)\b",
    # Delimiter / Role spoofing
    r"(<\|im_start\|>|<\|system\|>|\[system\]|###\s*system\s*instruction)",
]

_COMPILED_INJECTIONS = [
    re.compile(pat, re.IGNORECASE | re.DOTALL) for pat in _INJECTION_PATTERNS
]


def _normalize_text(text: str) -> str:
    """Strips invisible characters, normalizes Unicode lookalikes, and collapses whitespace."""
    # Convert fancy Unicode fonts (e.g., 𝐢𝐠𝐧𝐨𝐫𝐞) to standard ASCII equivalents
    normalized = unicodedata.normalize("NFKC", text)
    # Remove zero-width spaces / soft hyphens
    normalized = _INVISIBLE_CHARS_RE.sub("", normalized)
    # Collapse erratic spacing/newlines into single spaces
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized.strip()


def has_injection_pattern(text: str) -> bool:
    """Detects prompt injection attempts after normalizing obfuscation tricks."""
    if not text:
        return False

    cleaned = _normalize_text(text)
    return any(pattern.search(cleaned) for pattern in _COMPILED_INJECTIONS)
