"""
Flow: interview_prep_flow
Orchestrates the full interview preparation workflow:
  1. Collect candidate profile via user activity
  2. Generate tailored interview questions using IBM Granite
  3. Build a personalized preparation strategy
"""

from pydantic import BaseModel, Field
from ibm_watsonx_orchestrate.flow_builder.flows import Flow, flow, START, END
from ibm_watsonx_orchestrate.agent_builder.tools import tool, ToolPermission
import json
import requests
from typing import List, Optional


# ─── Shared Input Schema ─────────────────────────────────────────────────────

class InterviewPrepInput(BaseModel):
    """Input schema for the interview prep flow."""
    profile_name: str = Field(
        ...,
        description="Full name of the candidate"
    )
    job_role: str = Field(
        ...,
        description="Target job role e.g. 'Software Engineer', 'Data Scientist'"
    )
    experience_level: str = Field(
        ...,
        description="Experience level: entry, mid, or senior"
    )
    target_company_type: str = Field(
        default="general",
        description="Company type: startup, enterprise, big_tech, consulting, or general"
    )
    days_until_interview: int = Field(
        default=14,
        description="Number of days until the interview"
    )
    additional_context: Optional[str] = Field(
        default=None,
        description="Optional resume summary or job description context"
    )


class InterviewPrepOutput(BaseModel):
    """Output schema for the full interview prep flow result."""
    profile_name: str = Field(description="Candidate name")
    job_role: str = Field(description="Target job role")
    experience_level: str = Field(description="Experience level")
    questions_summary: str = Field(
        description="Formatted list of generated interview questions with tips"
    )
    preparation_strategy: str = Field(
        description="Formatted preparation strategy with daily plan and resources"
    )
    combined_report: str = Field(
        description="Full interview preparation report combining questions and strategy"
    )


# ─── IBM Granite Helper (self-contained) ────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
MODEL_ID = "ibm/granite-4-h-small"
PROJECT_ID = "27c077f3-15bb-4884-8bba-95b3064f5861"
API_KEY = "XYabwRKs9AnAkC_RGqClj7olprWvTmFO0x-wt3GwnUwT"


def _get_iam_token_flow() -> str:
    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}",
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["access_token"]


def _call_granite_flow(system_prompt: str, user_prompt: str) -> str:
    token = _get_iam_token_flow()
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "parameters": {"max_new_tokens": 2000, "temperature": 0.65},
    }
    resp = requests.post(
        WATSONX_URL,
        headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


# ─── Flow Tool: Run Full Interview Prep ──────────────────────────────────────

@tool(permission=ToolPermission.READ_ONLY)
def run_interview_prep_flow(input: InterviewPrepInput) -> InterviewPrepOutput:
    """
    Run the complete interview preparation workflow for a candidate.

    Generates tailored interview questions AND a preparation strategy using
    IBM Granite on watsonx.ai, combining both into a single comprehensive report.

    Args:
        input (InterviewPrepInput): Candidate profile including name, role,
            experience level, company type, days available, and optional context.

    Returns:
        InterviewPrepOutput: A full interview prep report with questions,
            tips, a daily preparation plan, resources, and confidence tips.
    """
    context_note = (
        f"\nExtra context: {input.additional_context}" if input.additional_context else ""
    )

    # ── Step 1: Generate Interview Questions ──────────────────────────────────
    questions_prompt = f"""
Generate tailored interview questions for:
- Candidate: {input.profile_name}
- Role: {input.job_role}
- Level: {input.experience_level}
{context_note}

Return JSON:
{{
  "questions": [
    {{"category": "technical|behavioral|situational", "question": "...", "tip": "..."}}
  ],
  "summary": "2-sentence encouragement message to {input.profile_name}"
}}

Include 4 technical, 3 behavioral, 2 situational, and 1 career-goal question.
"""

    raw_q = _call_granite_flow(
        "You are an expert interview coach. Return valid JSON only.",
        questions_prompt,
    )

    try:
        clean_q = raw_q.strip()
        if clean_q.startswith("```"):
            clean_q = clean_q.split("```")[1]
            if clean_q.startswith("json"):
                clean_q = clean_q[4:]
        q_data = json.loads(clean_q.strip())
    except Exception:
        q_data = {
            "questions": [
                {"category": "technical", "question": f"Describe your experience as a {input.job_role}.", "tip": "Use the STAR method."}
            ],
            "summary": f"Great start, {input.profile_name}! Let's prepare you well.",
        }

    # Format questions as readable text
    questions_text_parts = [f"# Interview Questions for {input.profile_name}\n"]
    questions_text_parts.append(f"**Role:** {input.job_role} | **Level:** {input.experience_level}\n\n")
    for i, q in enumerate(q_data.get("questions", []), 1):
        questions_text_parts.append(
            f"**Q{i} [{q.get('category', 'general').upper()}]:** {q.get('question', '')}\n"
            f"💡 *Tip:* {q.get('tip', '')}\n\n"
        )
    questions_text_parts.append(f"**Summary:** {q_data.get('summary', '')}\n")
    questions_summary = "".join(questions_text_parts)

    # ── Step 2: Build Preparation Strategy ───────────────────────────────────
    strategy_prompt = f"""
Create a {input.days_until_interview}-day interview preparation strategy for:
- Candidate: {input.profile_name}
- Role: {input.job_role}
- Level: {input.experience_level}
- Company Type: {input.target_company_type}

Return JSON:
{{
  "executive_summary": "...",
  "daily_plan": [{{"day": "Day 1-3", "tasks": ["...", "..."]}}],
  "key_topics_to_study": ["...", "..."],
  "recommended_resources": ["...", "..."],
  "confidence_boosters": ["...", "..."]
}}
"""

    raw_s = _call_granite_flow(
        "You are a career coach and interview strategist. Return valid JSON only.",
        strategy_prompt,
    )

    try:
        clean_s = raw_s.strip()
        if clean_s.startswith("```"):
            clean_s = clean_s.split("```")[1]
            if clean_s.startswith("json"):
                clean_s = clean_s[4:]
        s_data = json.loads(clean_s.strip())
    except Exception:
        s_data = {
            "executive_summary": f"A focused {input.days_until_interview}-day plan for {input.profile_name}.",
            "daily_plan": [{"day": "Week 1", "tasks": ["Research company", "Review resume", "Practice STAR stories"]},
                           {"day": "Week 2", "tasks": ["Mock interviews", "Prepare questions for interviewer", "Rest and review"]}],
            "key_topics_to_study": [f"{input.job_role} fundamentals", "STAR method", "Industry trends"],
            "recommended_resources": ["LinkedIn Learning", "Glassdoor", "Pramp.com"],
            "confidence_boosters": ["Practice aloud", "Visualise success", "Arrive/log in early"],
        }

    # Format strategy as readable text
    strategy_parts = [f"# Preparation Strategy for {input.profile_name}\n\n"]
    strategy_parts.append(f"**Overview:** {s_data.get('executive_summary', '')}\n\n")
    strategy_parts.append("## 📅 Daily Plan\n")
    for day_item in s_data.get("daily_plan", []):
        strategy_parts.append(f"\n**{day_item.get('day', '')}:**\n")
        for task in day_item.get("tasks", []):
            strategy_parts.append(f"- {task}\n")

    strategy_parts.append("\n## 📚 Key Topics to Study\n")
    for topic in s_data.get("key_topics_to_study", []):
        strategy_parts.append(f"- {topic}\n")

    strategy_parts.append("\n## 🔗 Recommended Resources\n")
    for res in s_data.get("recommended_resources", []):
        strategy_parts.append(f"- {res}\n")

    strategy_parts.append("\n## 💪 Confidence Boosters\n")
    for tip in s_data.get("confidence_boosters", []):
        strategy_parts.append(f"- {tip}\n")

    preparation_strategy = "".join(strategy_parts)

    # ── Step 3: Combine into Full Report ─────────────────────────────────────
    combined_report = (
        f"{'='*60}\n"
        f"  INTERVIEW PREPARATION REPORT\n"
        f"{'='*60}\n\n"
        f"{questions_summary}\n\n"
        f"{'─'*60}\n\n"
        f"{preparation_strategy}"
    )

    return InterviewPrepOutput(
        profile_name=input.profile_name,
        job_role=input.job_role,
        experience_level=input.experience_level,
        questions_summary=questions_summary,
        preparation_strategy=preparation_strategy,
        combined_report=combined_report,
    )


# ─── Flow Definition ──────────────────────────────────────────────────────────

@flow(
    name="interview_prep_flow",
    display_name="Interview Preparation Flow",
    description=(
        "Generates tailored interview questions and a personalised preparation "
        "strategy for a candidate using IBM Granite on watsonx.ai"
    ),
    input_schema=InterviewPrepInput,
)
def build_interview_prep_flow(aflow: Flow) -> Flow:
    """
    Build the interview preparation orchestration flow.

    The flow invokes the run_interview_prep_flow tool which calls IBM Granite
    twice (once for questions, once for strategy) and returns a combined report.
    """
    prep_node = aflow.tool(run_interview_prep_flow)
    aflow.sequence(START, prep_node, END)
    return aflow
