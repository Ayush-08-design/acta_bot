import asyncio
import os
from datetime import datetime

from celery import Celery
from deepeval.metrics import (
    AnswerRelevancyMetric,
    BiasMetric,
    ContextualRelevancyMetric,
    FaithfulnessMetric,
    ToxicityMetric,
)
from deepeval.test_case import LLMTestCase

from .eval_logger import logger
from .eval_model import async_judge

celery_app = Celery(
    "eval_workers",
    broker=os.getenv("REDIS_URL", "redis://localhost:6379/0"),
    backend=os.getenv("REDIS_URL", "redis://localhost:6379/0")
)

async def evaluate_and_save(payload: dict):
    """Executes LLM-as-a-judge metrics and saves results to a Json file."""
    
    normal_test_case = LLMTestCase(
        input=payload["user_query"],
        actual_output=payload["agent_response"],
    )
    relevance = AnswerRelevancyMetric(threshold=0.7, model=async_judge)
    
    evals = [{'metric' : relevance , 'test_case' : normal_test_case}]
    
    if payload['is_rag']:
        rag_case = LLMTestCase(
            input=payload["user_query"],
            actual_output=payload["agent_response"],
            retrieval_context=payload["retrieved_chunks"] 
        )
        faithfulness = FaithfulnessMetric(threshold=0.7, model=async_judge)
        context_relevance = ContextualRelevancyMetric(threshold=0.6, model=async_judge)
        
        evals.append({'metric' : faithfulness , 'test_case' : rag_case})
        evals.append({'metric' : context_relevance , 'test_case' : rag_case})
        
    bias = BiasMetric(threshold=0.7, model=async_judge)
    toxicity = ToxicityMetric(threshold=0.7, model=async_judge)
    
    evals.append({'metric' : bias , 'test_case' : normal_test_case})
    evals.append({'metric' : toxicity , 'test_case' : normal_test_case})
    
    tasks = [metric['metric'].a_measure(metric['test_case']) for metric in evals]
    
    await asyncio.gather(*tasks)
    
    eval_record = {
        "trace_id": payload["trace_id"],
        "session_id": payload.get("session_id"),
        "timestamp": datetime.now(),
        "metrics": {
            "faithfulness": {
                "score": float(faithfulness.score),
                "passed": bool(faithfulness.is_successful()),
                "reason": faithfulness.reason
            },
            "answer_relevance": {
                "score": float(relevance.score),
                "passed": bool(relevance.is_successful()),
                "reason": relevance.reason
            }
        },
        "metadata": payload.get("metadata", {})
    }
    
    logger.add_item(eval_record)
    

@celery_app.task(name="tasks.process_live_evaluation")
def process_live_evaluation(payload: dict):
    """Sync wrapper wrapper for Celery to handle async event loop execution."""
    asyncio.run(evaluate_and_save(payload))
