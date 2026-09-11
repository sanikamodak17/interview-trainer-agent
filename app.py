"""
Backend API for Interview Trainer Agent
FastAPI server that connects the frontend to IBM Granite on watsonx.ai.

Start: uvicorn app:app --reload --port 8000
"""

import json
import requests
import os 
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ─── App Setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Interview Trainer Agent API",
    description="Backend for the IBM Granite-powered Interview Trainer Agent",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── IBM Granite Configuration ────────────────────────────────────────────────

WATSONX_URL = "https://us-south.ml.cloud.ibm.com/ml/v1/text/chat?version=2023-05-29"
MODEL_ID = "ibm/granite-4-h-small"
PROJECT_ID = "27c077f3-15bb-4884-8bba-95b3064f5861"
# API key is read from the environment variable IBM_API_KEY first;
# the hardcoded value below is used as a fallback for local dev.
API_KEY = os.environ.get("IBM_API_KEY", "XYabwRKs9AnAkC_RGqClj7olprWvTmFO0x-wt3GwnUwT")


_iam_token_cache: Dict[str, Any] = {}


def get_iam_token() -> str:
    """Fetch (and cache) an IBM Cloud IAM bearer token."""
    import time
    now = time.time()
    if _iam_token_cache.get("token") and now < _iam_token_cache.get("expires_at", 0):
        return _iam_token_cache["token"]

    resp = requests.post(
        "https://iam.cloud.ibm.com/identity/token",
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}",
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    _iam_token_cache["token"] = data["access_token"]
    _iam_token_cache["expires_at"] = now + data.get("expires_in", 3600) - 60
    return _iam_token_cache["token"]


def call_granite(system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    """Call IBM Granite on watsonx.ai and return the response text."""
    token = get_iam_token()
    payload = {
        "model_id": MODEL_ID,
        "project_id": PROJECT_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": 0.65,
        },
    }
    resp = requests.post(
        WATSONX_URL,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        json=payload,
        timeout=90,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"]


def parse_json_response(raw: str, fallback: dict) -> dict:
    """Safely parse a JSON string from an LLM response."""
    try:
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        return json.loads(clean.strip())
    except Exception:
        return fallback


# ─── Request/Response Schemas ─────────────────────────────────────────────────

class GenerateQuestionsRequest(BaseModel):
    profile_name: str = Field(..., description="Candidate's name")
    job_role: str = Field(..., description="Target job role")
    experience_level: str = Field(default="mid", description="entry, mid, or senior")
    additional_context: Optional[str] = Field(default=None)


class InterviewQuestion(BaseModel):
    category: str
    question: str
    tip: str


class GenerateQuestionsResponse(BaseModel):
    profile_name: str
    job_role: str
    experience_level: str
    questions: List[InterviewQuestion]
    summary: str


class EvaluateAnswerRequest(BaseModel):
    interview_question: str = Field(..., description="The interview question")
    candidate_answer: str = Field(..., description="Candidate's answer")
    job_role: str = Field(default="Software Engineer")
    experience_level: str = Field(default="mid")


class EvaluateAnswerResponse(BaseModel):
    score: int
    strengths: str
    areas_for_improvement: str
    model_answer: str
    key_takeaway: str


class BuildStrategyRequest(BaseModel):
    profile_name: str
    job_role: str
    experience_level: str = Field(default="mid")
    target_company_type: str = Field(default="general")
    days_until_interview: int = Field(default=14)
    weak_areas: Optional[str] = Field(default=None)


class DailyTask(BaseModel):
    day: str
    tasks: List[str]


class BuildStrategyResponse(BaseModel):
    profile_name: str
    job_role: str
    target_company_type: str
    executive_summary: str
    daily_plan: List[DailyTask]
    key_topics_to_study: List[str]
    recommended_resources: List[str]
    confidence_boosters: List[str]


class FullReportRequest(BaseModel):
    profile_name: str
    job_role: str
    experience_level: str = Field(default="mid")
    target_company_type: str = Field(default="general")
    days_until_interview: int = Field(default=14)
    additional_context: Optional[str] = Field(default=None)


class FullReportResponse(BaseModel):
    profile_name: str
    job_role: str
    experience_level: str
    questions_summary: str
    preparation_strategy: str
    combined_report: str


class ChatRequest(BaseModel):
    message: str
    history: Optional[List[Dict[str, str]]] = Field(default_factory=list)


class ChatResponse(BaseModel):
    response: str


# ─── Health Check ─────────────────────────────────────────────────────────────

@app.get("/")
async def health_check():
    return {
        "status": "ok",
        "service": "Interview Trainer Agent API",
        "model": MODEL_ID,
        "project_id": PROJECT_ID,
    }


@app.get("/health/ibm")
async def ibm_health():
    """Verify IBM Cloud API key and watsonx.ai reachability."""
    try:
        resp = requests.post(
            "https://iam.cloud.ibm.com/identity/token",
            headers={"Content-Type": "application/x-www-form-urlencoded"},
            data=f"grant_type=urn:ibm:params:oauth:grant-type:apikey&apikey={API_KEY}",
            timeout=15,
        )
        if resp.status_code == 200:
            token_data = resp.json()
            return {
                "iam": "ok",
                "token_type": token_data.get("token_type"),
                "expires_in": token_data.get("expires_in"),
                "model": MODEL_ID,
                "project_id": PROJECT_ID,
                "api_key_source": "env:IBM_API_KEY" if os.environ.get("IBM_API_KEY") else "hardcoded_fallback",
            }
        else:
            body = resp.json()
            return {
                "iam": "error",
                "http_status": resp.status_code,
                "error_code": body.get("errorCode"),
                "error_message": body.get("errorMessage"),
                "fix": (
                    "Go to https://cloud.ibm.com/iam/apikeys, create a new API key, "
                    "then restart the server with: IBM_API_KEY=<new-key> uvicorn app:app --port 8000"
                ),
            }
    except Exception as e:
        return {"iam": "unreachable", "error": str(e)}


# ─── Endpoint: Generate Interview Questions ───────────────────────────────────

@app.post("/generate-questions", response_model=GenerateQuestionsResponse)
async def generate_interview_questions(req: GenerateQuestionsRequest):
    """Generate tailored interview questions using IBM Granite."""
    context = f"\nAdditional context: {req.additional_context}" if req.additional_context else ""
    user_prompt = f"""
Generate interview questions for:
- Name: {req.profile_name}
- Role: {req.job_role}
- Level: {req.experience_level}
{context}

Return JSON:
{{
  "questions": [
    {{"category": "technical|behavioral|situational", "question": "...", "tip": "..."}}
  ],
  "summary": "2-sentence encouragement addressed to {req.profile_name}"
}}

Include: 4 technical, 3 behavioral, 2 situational, 1 career-goal question.
All appropriate for {req.experience_level} level {req.job_role}.
"""
    try:
        raw = call_granite(
            "You are an expert interview coach. Return valid JSON only.",
            user_prompt,
            max_tokens=1500,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite call failed: {str(e)}")

    fallback = {
        "questions": [
            {"category": "technical", "question": f"Describe your experience as a {req.job_role}.", "tip": "Use the STAR method."},
        ],
        "summary": f"Great start, {req.profile_name}! Let's prepare you well.",
    }
    data = parse_json_response(raw, fallback)

    questions = [
        InterviewQuestion(
            category=q.get("category", "general"),
            question=q.get("question", ""),
            tip=q.get("tip", ""),
        )
        for q in data.get("questions", [])
    ]

    return GenerateQuestionsResponse(
        profile_name=req.profile_name,
        job_role=req.job_role,
        experience_level=req.experience_level,
        questions=questions,
        summary=data.get("summary", f"Best of luck in your {req.job_role} interview, {req.profile_name}!"),
    )


# ─── Endpoint: Evaluate Interview Answer ──────────────────────────────────────

@app.post("/evaluate-answer", response_model=EvaluateAnswerResponse)
async def evaluate_interview_answer(req: EvaluateAnswerRequest):
    """Score and provide detailed feedback on a candidate's interview answer."""
    user_prompt = f"""
Evaluate this interview answer:
Role: {req.job_role} | Level: {req.experience_level}
Question: {req.interview_question}
Answer: {req.candidate_answer}

Return JSON:
{{
  "score": <1-10>,
  "strengths": "<2-3 sentences on what was done well>",
  "areas_for_improvement": "<2-3 specific actionable improvements>",
  "model_answer": "<complete polished model answer for {req.experience_level} {req.job_role}>",
  "key_takeaway": "<single most important improvement tip>"
}}

Scoring: 9-10=excellent, 7-8=good, 5-6=average, 3-4=below average, 1-2=poor
"""
    try:
        raw = call_granite(
            "You are a senior hiring manager evaluating interview answers. Return valid JSON only.",
            user_prompt,
            max_tokens=1000,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite call failed: {str(e)}")

    fallback = {
        "score": 5,
        "strengths": "You attempted to address the question.",
        "areas_for_improvement": "Use the STAR method and include specific metrics.",
        "model_answer": f"A strong {req.experience_level} {req.job_role} would provide a concrete example with measurable results.",
        "key_takeaway": "Always quantify your impact with numbers.",
    }
    data = parse_json_response(raw, fallback)

    return EvaluateAnswerResponse(
        score=int(data.get("score", 5)),
        strengths=data.get("strengths", ""),
        areas_for_improvement=data.get("areas_for_improvement", ""),
        model_answer=data.get("model_answer", ""),
        key_takeaway=data.get("key_takeaway", ""),
    )


# ─── Endpoint: Build Preparation Strategy ────────────────────────────────────

@app.post("/build-strategy", response_model=BuildStrategyResponse)
async def build_preparation_strategy(req: BuildStrategyRequest):
    """Generate a personalised day-by-day preparation plan."""
    weak_section = f"\nWeak areas: {req.weak_areas}" if req.weak_areas else ""
    user_prompt = f"""
Create a {req.days_until_interview}-day interview preparation strategy for:
- Name: {req.profile_name}
- Role: {req.job_role}
- Level: {req.experience_level}
- Company Type: {req.target_company_type}
{weak_section}

Return JSON:
{{
  "executive_summary": "2-3 sentence overview addressed to {req.profile_name}",
  "daily_plan": [{{"day": "Day 1-3", "tasks": ["task1", "task2"]}}],
  "key_topics_to_study": ["topic1", "topic2", "topic3", "topic4", "topic5"],
  "recommended_resources": ["resource1", "resource2", "resource3", "resource4"],
  "confidence_boosters": ["tip1", "tip2", "tip3"]
}}

Create {min(req.days_until_interview, 7)} milestones in daily_plan.
"""
    try:
        raw = call_granite(
            "You are a world-class career coach. Return valid JSON only.",
            user_prompt,
            max_tokens=1500,
        )
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite call failed: {str(e)}")

    fallback = {
        "executive_summary": f"A focused {req.days_until_interview}-day plan for {req.profile_name}.",
        "daily_plan": [
            {"day": "Week 1", "tasks": ["Research company", "Review resume", "Practice STAR stories"]},
            {"day": "Week 2", "tasks": ["Mock interviews", "Prepare questions for interviewer", "Rest and review"]},
        ],
        "key_topics_to_study": [f"{req.job_role} fundamentals", "STAR method", "Industry trends"],
        "recommended_resources": ["LinkedIn Learning", "Glassdoor", "Pramp.com"],
        "confidence_boosters": ["Practice aloud", "Visualise success", "Arrive/log in early"],
    }
    data = parse_json_response(raw, fallback)

    daily_plan = [
        DailyTask(day=item.get("day", ""), tasks=item.get("tasks", []))
        for item in data.get("daily_plan", [])
    ]

    return BuildStrategyResponse(
        profile_name=req.profile_name,
        job_role=req.job_role,
        target_company_type=req.target_company_type,
        executive_summary=data.get("executive_summary", ""),
        daily_plan=daily_plan,
        key_topics_to_study=data.get("key_topics_to_study", []),
        recommended_resources=data.get("recommended_resources", []),
        confidence_boosters=data.get("confidence_boosters", []),
    )


# ─── Endpoint: Full Prep Report ───────────────────────────────────────────────

@app.post("/full-report", response_model=FullReportResponse)
async def full_report(req: FullReportRequest):
    """Generate a comprehensive prep report: questions + strategy combined."""
    context = f"\nContext: {req.additional_context}" if req.additional_context else ""

    # ── Questions ──
    q_prompt = f"""
Generate interview questions for {req.profile_name}, {req.experience_level} {req.job_role}.
{context}
Return JSON: {{"questions": [{{"category": "...", "question": "...", "tip": "..."}}], "summary": "..."}}
Include 4 technical, 3 behavioral, 2 situational, 1 career-goal.
"""
    try:
        raw_q = call_granite("You are an interview coach. Return JSON only.", q_prompt, 1500)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite error (questions): {str(e)}")

    q_data = parse_json_response(raw_q, {
        "questions": [{"category": "technical", "question": f"Describe your {req.job_role} experience.", "tip": "Use STAR."}],
        "summary": f"Best of luck, {req.profile_name}!",
    })

    q_parts = [f"# Interview Questions for {req.profile_name}\n\n"]
    q_parts.append(f"**Role:** {req.job_role} | **Level:** {req.experience_level}\n\n")
    for i, q in enumerate(q_data.get("questions", []), 1):
        q_parts.append(f"**Q{i} [{q.get('category','general').upper()}]:** {q.get('question','')}\n")
        q_parts.append(f"💡 *Tip:* {q.get('tip','')}\n\n")
    q_parts.append(f"**Summary:** {q_data.get('summary', '')}\n")
    questions_summary = "".join(q_parts)

    # ── Strategy ──
    s_prompt = f"""
Create a {req.days_until_interview}-day prep strategy for {req.profile_name},
{req.experience_level} {req.job_role}, targeting {req.target_company_type} company.
Return JSON: {{"executive_summary":"...","daily_plan":[{{"day":"...","tasks":["..."]}}],"key_topics_to_study":["..."],"recommended_resources":["..."],"confidence_boosters":["..."]}}
"""
    try:
        raw_s = call_granite("You are a career coach. Return JSON only.", s_prompt, 1500)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite error (strategy): {str(e)}")

    s_data = parse_json_response(raw_s, {
        "executive_summary": f"Your {req.days_until_interview}-day plan.",
        "daily_plan": [{"day": "Week 1", "tasks": ["Research", "Practice", "Mock interview"]}],
        "key_topics_to_study": [f"{req.job_role} skills", "STAR method"],
        "recommended_resources": ["Glassdoor", "LinkedIn Learning"],
        "confidence_boosters": ["Practice aloud", "Rest well the night before"],
    })

    s_parts = [f"# Preparation Strategy for {req.profile_name}\n\n"]
    s_parts.append(f"**Overview:** {s_data.get('executive_summary','')}\n\n")
    s_parts.append("## Daily Plan\n")
    for d in s_data.get("daily_plan", []):
        s_parts.append(f"\n**{d.get('day','')}:**\n")
        for t in d.get("tasks", []):
            s_parts.append(f"- {t}\n")
    s_parts.append("\n## Key Topics\n")
    for t in s_data.get("key_topics_to_study", []):
        s_parts.append(f"- {t}\n")
    s_parts.append("\n## Resources\n")
    for r in s_data.get("recommended_resources", []):
        s_parts.append(f"- {r}\n")
    s_parts.append("\n## Confidence Tips\n")
    for c in s_data.get("confidence_boosters", []):
        s_parts.append(f"- {c}\n")
    preparation_strategy = "".join(s_parts)

    combined = (
        "=" * 60 + "\n"
        "  INTERVIEW PREPARATION REPORT\n"
        + "=" * 60 + "\n\n"
        + questions_summary + "\n\n"
        + "-" * 60 + "\n\n"
        + preparation_strategy
    )

    return FullReportResponse(
        profile_name=req.profile_name,
        job_role=req.job_role,
        experience_level=req.experience_level,
        questions_summary=questions_summary,
        preparation_strategy=preparation_strategy,
        combined_report=combined,
    )


# ─── Endpoint: General Chat ───────────────────────────────────────────────────

@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """General conversational chat with IBM Granite for interview coaching."""
    system = (
        "You are an expert AI interview coach powered by IBM Granite. "
        "Help candidates with interview preparation, career advice, resume tips, "
        "and interview strategies. Be encouraging, professional, and specific."
    )
    try:
        response = call_granite(system, req.message, max_tokens=800)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"IBM Granite call failed: {str(e)}")
    return ChatResponse(response=response)
