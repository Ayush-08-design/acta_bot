import logging

from src.app.llm.guardrails.pii_filter import redact_pii
from src.app.llm.guardrails.utils import run_llm_safety_check

logger = logging.getLogger(__name__)


async def enforce_output_guardrail(chunks: str) -> dict:

    # Deterministic PII scrubbing on chunks
    clean_output = redact_pii(chunks)

    # LLM compliance check on intermediate chunks
    verdict = await run_llm_safety_check(clean_output, stage="assistant output")
    if not verdict.is_safe:
        
        logger.debug(f"Unsafe response detected: [{chunks}], Reason: [{verdict.reason}]")
        return {
            "blocked": True,
            "block_reason": verdict.reason,
            "chunks": "I cannot share that response due to safety and compliance policies.",
        }

    return {"blocked": False, "chunks": clean_output}
