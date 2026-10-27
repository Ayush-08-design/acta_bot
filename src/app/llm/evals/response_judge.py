from dataclasses import dataclass

from langchain_ollama import ChatOllama


@dataclass
class JudgeResult:
    '''Result of an LLM judge Evaluation'''
    crieterion : str
    score : int
    max_score : int
    reasoning : str

RUBRICS = {
    'relevence' : {
        'description' : "Does the Response directly address the user's question",
        'levels' : {
            1: "Completely off-topic or addresses a different question entirely.",
            2: "Tangentially related but misses the core question.",
            3: "Addresses the question but includes significant irrelevant content.",
            4: "Directly addresses the question with minor tangents.",
            5: "Precisely and completely addresses the question asked.",
        },
    },
    "accuracy": {
        "description": "Is the factual content of the response correct?",
        "levels": {
            1: "Contains critical factual errors that would mislead the reader.",
            2: "Multiple factual errors on important points.",
            3: "Mostly accurate but contains one notable error.",
            4: "Accurate with only trivial imprecisions.",
            5: "Completely accurate with no factual errors.",
        },
    },
    "completeness": {
        "description": "Does the response cover all important aspects of the question?",
        "levels": {
            1: "Addresses less than 20 percent of what the question requires.",
            2: "Covers some aspects but misses major required components.",
            3: "Covers the basics but lacks depth on important points.",
            4: "Comprehensive coverage with minor gaps.",
            5: "Thoroughly covers all aspects the question requires.",
        },
    },
}


class LLMJudge:

    def __init__(self, model: str = "qwen2.5:0.5b"):
        self.model = ChatOllama(model = model).with_structured_output(JudgeResult)

    def evaluate(
        self, question: str, response: str, criterion: str
    ) -> JudgeResult:
        """Evaluate a single response on a single criterion."""
        rubric = RUBRICS[criterion]
        levels_text = "\n".join(
            f"Score {score}: {desc}"
            for score, desc in rubric["levels"].items()
        )
        judge_prompt = f"""You are an expert evaluator. Your job is to score an AI assistant's response.

        CRITERION: {rubric['description']}

        SCORING RUBRIC:
        {levels_text}

        USER QUESTION:
        {question}

        AI RESPONSE:
        {response}

        Evaluate the response on the criterion above. You must respond with valid JSON only:
        {{"score": <integer 1-5>, "reasoning": "<2-3 sentence explanation>"}}"""

        judge_response = self.model.invoke(
            input=[{"role": "user", "content": judge_prompt}],
            temperature=0.0,
        )
        return judge_response

    def evaluate_all(
        self, question: str, response: str, criteria: list[str] | None = None
    ) -> list[JudgeResult]:
        """Evaluate a response across all specified criteria."""
        criteria = criteria or list(RUBRICS.keys())
        return [self.evaluate(question, response, c) for c in criteria]
