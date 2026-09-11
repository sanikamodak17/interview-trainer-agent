"""
Tool: generate_interview_questions
Generates tailored interview questions using IBM Granite via watsonx.ai,
based on the user's job role and experience level.
"""

import json
import requests
from typing import Optional, List
from pydantic import BaseModel, Field
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class InterviewQuestionsInput(BaseModel):
    """Input schema for generating interview questions."""
    profile_name: str = Field(
        ...,
        description="Full name of the candidate for personalized output"
    )
    job_role: str = Field(
        ...,
        description="Target job role or title, e.g. 'Software Engineer', 'Data Scientist'"
    )
    experience_level: str = Field(
        ...,
        description="Candidate's experience level: 'entry', 'mid', or 'senior'"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Optional: extra context from resume or job description"
    )


class InterviewQuestion(BaseModel):
    """A single interview question with category and tips."""
    category: str = Field(description="Category: technical, behavioral, or situational")
    question: str = Field(description="The interview question")
    tip: str = Field(description="Short preparation tip or model answer guidance")


class InterviewQuestionsOutput(BaseModel):
    """Output schema containing generated interview questions."""
    profile_name: str = Field(description="Candidate name")
    job_role: str = Field(description="Target job role")
    experience_level: str = Field(description="Experience level")
    questions: List[InterviewQuestion] = Field(
        description="List of tailored interview questions with tips"
    )
    summary: str = Field(description="Overall preparation summary for the candidate")


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
            "max_new_tokens": 1500,
            "temperature": 0.7,
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
def generate_interview_questions(input: InterviewQuestionsInput) -> InterviewQuestionsOutput:
    """
    Generate tailored interview questions for a candidate using IBM Granite.

    Uses the IBM Granite model on watsonx.ai to produce role-specific and
    experience-level-appropriate interview questions with preparation tips.

    Args:
        input (InterviewQuestionsInput): Candidate profile including name,
            job role, experience level, and optional resume context.

    Returns:
        InterviewQuestionsOutput: A structured set of interview questions
            with categories, tips, and a preparation summary.
    """
    system_prompt = (
        "You are an expert interview coach and career advisor. "
        "Your task is to generate targeted, relevant, and high-quality interview "
        "questions for job candidates based on their profile. "
        "Always return your response as a valid JSON object matching the schema provided."
    )

    context_section = (
        f"\nAdditional context from resume/JD: {input.additional_context}"
        if input.additional_context
        else ""
    )

    user_prompt = f"""
Generate a comprehensive set of interview questions for this candidate:
- Name: {input.profile_name}
- Target Role: {input.job_role}
- Experience Level: {input.experience_level}
{context_section}

Return a JSON object with this exact structure:
{{
  "questions": [
    {{
      "category": "technical|behavioral|situational",
      "question": "The interview question text",
      "tip": "A short preparation tip or model answer guidance (1-2 sentences)"
    }}
  ],
  "summary": "A 2-3 sentence overall preparation summary addressing {input.profile_name} directly"
}}

Include:
- 4 technical questions appropriate for {input.experience_level} level {input.job_role}
- 3 behavioral questions (STAR method based)
- 2 situational/scenario-based questions
- 1 question about career goals and motivation

Ensure questions match the {input.experience_level} experience level accurately.
"""

    raw_response = _call_granite(system_prompt, user_prompt)

    # Parse JSON from the model response
    try:
        # Strip markdown fences if present
        clean = raw_response.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        data = json.loads(clean.strip())
    except Exception:
        data = {
            "questions": [
                {
                    "category": "technical",
                    "question": f"Describe your experience as a {input.job_role}.",
                    "tip": "Use specific examples from past roles."
                }
            ],
            "summary": f"Hi {input.profile_name}! Focus on your {input.job_role} experience and prepare STAR stories.",
        }

    questions = [
        InterviewQuestion(
            category=q.get("category", "general"),
            question=q.get("question", ""),
            tip=q.get("tip", ""),
        )
        for q in data.get("questions", [])
    ]

    return InterviewQuestionsOutput(
        profile_name=input.profile_name,
        job_role=input.job_role,
        experience_level=input.experience_level,
        questions=questions,
        summary=data.get("summary", f"Best of luck in your {input.job_role} interview, {input.profile_name}!"),
    )
