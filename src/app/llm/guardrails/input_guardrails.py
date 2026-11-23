from src.app.llm.guardrails.pii_filter import has_injection_pattern, redact_pii
from src.app.llm.guardrails.utils import run_llm_safety_check


async def enforce_input_guardrail(input_text: str) -> dict:

    # Fast deterministic regex check
    if has_injection_pattern(input_text):
        return {
            "blocked": True,
            "block_reason": "Blocked by input policy (prompt injection detected).",
            "messages": "I cannot process that request.",
        }

    # Redact user PII before the main LLM processes it
    clean_text = redact_pii(input_text)
    input_text = clean_text  # overwrite input message with safe msg

    # Fast LLM safety & topic check
    verdict = await run_llm_safety_check(clean_text, stage="user input")
    if not verdict.is_safe:
        return {
            "blocked": True,
            "block_reason": verdict.reason,
            "messages": "Sorry, I cannot help you with given query",
        }

    return {"blocked": False, "block_reason": "", "messages": input_text}
