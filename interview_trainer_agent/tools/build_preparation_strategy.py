"""
Tool: build_preparation_strategy
Generates a personalised, structured interview preparation strategy
for a candidate using IBM Granite on watsonx.ai.
"""

import json
import requests
from typing import List
from pydantic import BaseModel, Field
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission


# ─── Pydantic Models ─────────────────────────────────────────────────────────

class PrepStrategyInput(BaseModel):
    """Input for building a personalised interview preparation strategy."""
    profile_name: str = Field(
        ...,
        description="Full name of the candidate"
    )
    job_role: str = Field(
        ...,
        description="Target job role, e.g. 'Product Manager', 'Data Engineer'"
    )
    experience_level: str = Field(
        ...,
        description="Experience level: entry, mid, or senior"
    )
    target_company_type: str = Field(
        default="general",
        description="Type of company: 'startup', 'enterprise', 'big_tech', 'consulting', or 'general'"
    )
    days_until_interview: int = Field(
        default=14,
        description="Number of days the candidate has to prepare"
    )
    weak_areas: str = Field(
        default="",
        description="Optional: areas the candidate feels less confident about"
    )


class DailyTask(BaseModel):
    """A single daily preparation task."""
    day: str = Field(description="Day or date range label, e.g. 'Day 1', 'Week 1'")
    tasks: List[str] = Field(description="List of preparation tasks for that day/period")


class PrepStrategyOutput(BaseModel):
    """Output containing the complete preparation strategy."""
    profile_name: str = Field(description="Candidate name")
    job_role: str = Field(description="Target job role")
    target_company_type: str = Field(description="Company type being targeted")
    executive_summary: str = Field(
        description="High-level summary of the preparation strategy"
    )
    daily_plan: List[DailyTask] = Field(
        description="Day-by-day or week-by-week preparation schedule"
    )
    key_topics_to_study: List[str] = Field(
        description="Priority topics to study for the specific role"
    )
    recommended_resources: List[str] = Field(
        description="Books, websites, and tools recommended for preparation"
    )
    confidence_boosters: List[str] = Field(
        description="Tips to build confidence and reduce interview anxiety"
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
            "max_new_tokens": 1500,
            "temperature": 0.6,
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
def build_preparation_strategy(input: PrepStrategyInput) -> PrepStrategyOutput:
    """
    Build a personalised interview preparation strategy for a candidate.

    Uses IBM Granite to generate a structured, day-by-day preparation plan
    tailored to the candidate's role, experience level, company type, and
    available preparation time.

    Args:
        input (PrepStrategyInput): Candidate profile including name, role,
            experience level, target company type, days available, and weak areas.

    Returns:
        PrepStrategyOutput: A complete preparation strategy with a daily plan,
            key topics, recommended resources, and confidence-building tips.
    """
    system_prompt = (
        "You are a world-class career coach and interview strategist. "
        "You create highly personalised and actionable interview preparation plans "
        "based on the candidate's profile, target role, and timeline. "
        "Always return your response as a valid JSON object."
    )

    weak_section = (
        f"\nAreas the candidate wants to strengthen: {input.weak_areas}"
        if input.weak_areas
        else ""
    )

    user_prompt = f"""
Create a detailed interview preparation strategy for:
- Candidate: {input.profile_name}
- Target Role: {input.job_role}
- Experience Level: {input.experience_level}
- Target Company Type: {input.target_company_type}
- Days to Prepare: {input.days_until_interview}
{weak_section}

Return a JSON object with this exact structure:
{{
  "executive_summary": "<2-3 sentence overview of the strategy for {input.profile_name}>",
  "daily_plan": [
    {{
      "day": "Day 1-3",
      "tasks": ["task 1", "task 2", "task 3"]
    }}
  ],
  "key_topics_to_study": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "recommended_resources": ["resource1", "resource2", "resource3", "resource4"],
  "confidence_boosters": ["tip1", "tip2", "tip3"]
}}

For the daily_plan, create {min(input.days_until_interview, 7)} meaningful milestones 
appropriate for preparing for a {input.experience_level} {input.job_role} role at a {input.target_company_type} company.
Include at least 5 key_topics_to_study and 4 recommended_resources.
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
            "executive_summary": f"Hi {input.profile_name}! Here is your tailored prep plan for a {input.job_role} role.",
            "daily_plan": [
                {
                    "day": "Week 1",
                    "tasks": [
                        "Research the company and role",
                        "Review your resume and prepare STAR stories",
                        "Practice common behavioral questions",
                    ],
                },
                {
                    "day": "Week 2",
                    "tasks": [
                        "Do mock interviews",
                        "Prepare questions for the interviewer",
                        "Rest and review key points the day before",
                    ],
                },
            ],
            "key_topics_to_study": [
                f"Core {input.job_role} competencies",
                "STAR behavioral framework",
                "Industry trends",
                "Company products and competitors",
                "Salary negotiation",
            ],
            "recommended_resources": [
                "Cracking the Coding Interview (for tech roles)",
                "LinkedIn Learning interview courses",
                "Pramp.com for mock interviews",
                "Glassdoor for company insights",
            ],
            "confidence_boosters": [
                "Practice your answers aloud in front of a mirror",
                "Visualize a successful interview the morning of",
                "Remember: the interview is a two-way conversation",
            ],
        }

    daily_plan = [
        DailyTask(
            day=item.get("day", ""),
            tasks=item.get("tasks", []),
        )
        for item in data.get("daily_plan", [])
    ]

    return PrepStrategyOutput(
        profile_name=input.profile_name,
        job_role=input.job_role,
        target_company_type=input.target_company_type,
        executive_summary=data.get("executive_summary", ""),
        daily_plan=daily_plan,
        key_topics_to_study=data.get("key_topics_to_study", []),
        recommended_resources=data.get("recommended_resources", []),
        confidence_boosters=data.get("confidence_boosters", []),
    )
