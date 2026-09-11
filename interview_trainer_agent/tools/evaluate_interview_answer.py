"""
Tool: evaluate_interview_answer
Evaluates a candidate's answer to an interview question using IBM Granite,
providing a score, strengths, weaknesses, and an improved model answer.
"""

import json
import requests
from pydantic import BaseModel, Field
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class EvaluateAnswerInput(BaseModel):
    """Input for evaluating a candidate's interview answer."""
    interview_question: str = Field(
        ...,
        description="The interview question that was asked"
    )
    candidate_answer: str = Field(
        ...,
        description="The candidate's answer to evaluate"
    )
    job_role: str = Field(
        ...,
        description="The target job role for context"
    )
    experience_level: str = Field(
        default="mid",
        description="Candidate experience level: entry, mid, or senior"
    )


class EvaluateAnswerOutput(BaseModel):
    """Output containing the evaluation results of a candidate's answer."""
    score: int = Field(
        description="Score from 1 to 10 rating the quality of the answer"
    )
    strengths: str = Field(
        description="What the candidate did well in their answer"
    )
    areas_for_improvement: str = Field(
        description="Specific areas where the answer could be strengthened"
    )
    model_answer: str = Field(
        description="A polished model answer that demonstrates best practices"
    )
    key_takeaway: str = Field(
        description="The single most important improvement tip for the candidate"
    )


# ─── IBM Granite watsonx.ai Helper ───────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
MODEL_ID = "ibm/granite-4-h-small"
PROJECT_ID = "27c077f3-15bb-4884-8bba-95b3064f5861"
API_KEY = "XYabwRKs9AnAkC_RGqClj7olprWvTmFO0x-wt3GwnUwT"


def _get_iam_token() -> str:
    """Retrieve an IAM bearer token from IBM Cloud using the API key."""
    response = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}",
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["access_token"]


def _call_granite(system_prompt: str, user_prompt: str) -> str:
    """Call IBM Granite model via watsonx.ai chat endpoint."""
    token = _get_iam_token()
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "parameters": {
            "max_new_tokens": 1000,
            "temperature": 0.5,
        },
    }
    response = requests.post(
        WATSONX_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=60,
    )
    response.raise_for_status()
    result = response.json()
    return result["choices"][0]["message"]["content"]


# ─── Tool Definition ──────────────────────────────────────────────────────────

@tool(permission=ToolPermission.READ_ONLY)
def evaluate_interview_answer(input: EvaluateAnswerInput) -> EvaluateAnswerOutput:
    """
    Evaluate a candidate's interview answer and provide actionable feedback.

    Uses IBM Granite on watsonx.ai to score the answer, identify strengths
    and weaknesses, and generate a polished model answer for comparison.

    Args:
        input (EvaluateAnswerInput): The interview question, the candidate's
            answer, the target job role, and the experience level.

    Returns:
        EvaluateAnswerOutput: Score (1-10), strengths, areas for improvement,
            a model answer, and a key takeaway tip.
    """
    system_prompt = (
        "You are a senior hiring manager and interview coach with 20 years of experience. "
        "Your role is to give honest, constructive, and detailed feedback on interview answers. "
        "Always return your response as a valid JSON object."
    )

    user_prompt = f"""
Evaluate the following interview answer:

Job Role: {input.job_role}
Experience Level: {input.experience_level}
Question: {input.interview_question}
Candidate's Answer: {input.candidate_answer}

Return a JSON object with this exact structure:
{{
  "score": <integer from 1 to 10>,
  "strengths": "<What the candidate did well - 2-3 sentences>",
  "areas_for_improvement": "<Specific, actionable improvements - 2-3 sentences>",
  "model_answer": "<A complete, polished model answer demonstrating best practices for a {input.experience_level} {input.job_role}>",
  "key_takeaway": "<The single most important improvement tip in one sentence>"
}}

Scoring guide:
- 9-10: Excellent; clear, structured, specific with measurable results
- 7-8: Good; covers the key points but lacks depth or specifics
- 5-6: Average; partially addresses the question but misses key elements
- 3-4: Below average; vague, incomplete, or off-topic
- 1-2: Poor; does not address the question meaningfully
"""

    raw_response = _call_granite(system_prompt, user_prompt)

    try:
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        data = {
            "score": 5,
            "strengths": "You attempted to answer the question.",
            "areas_for_improvement": "Structure your answer using the STAR method and include specific examples.",
            "model_answer": f"A strong {input.experience_level} {input.job_role} would answer this by providing a concrete example with measurable outcomes.",
            "key_takeaway": "Always include specific metrics and results in your answers.",
        }

    return EvaluateAnswerOutput(
        score=int(data.get("score", 5)),
        strengths=data.get("strengths", ""),
        areas_for_improvement=data.get("areas_for_improvement", ""),
        model_answer=data.get("model_answer", ""),
        key_takeaway=data.get("key_takeaway", ""),
    )
