# 🎯 Interview Trainer Agent

An AI-powered interview preparation assistant built on **IBM watsonx Orchestrate** and **IBM Granite** (`ibm/granite-4-h-small`). It uses Retrieval-Augmented Generation (RAG) to provide candidates with tailored interview questions, answer evaluation with feedback, and personalised preparation strategies.

---

## Architecture Diagram

```mermaid
graph TB
    User[🧑 Candidate] -->|Chat| Agent[🤖 Interview Trainer Agent\nwatsonx Orchestrate]
    Agent -->|RAG Retrieval| KB[(📚 Interview\nKnowledge Base\nMilvus / Slate Embeddings)]
    Agent -->|Generate Questions| T1[🔧 generate_interview_questions\nIBM Granite]
    Agent -->|Evaluate Answers| T2[🔧 evaluate_interview_answer\nIBM Granite]
    Agent -->|Build Strategy| T3[🔧 build_preparation_strategy\nIBM Granite]
    Agent -->|Full Prep Report| F1[🔀 interview_prep_flow\nOrchestration Flow]
    F1 -->|Questions + Strategy| Granite[⚡ IBM Granite\nibm/granite-4-h-small\nwatsonx.ai]
    T1 --> Granite
    T2 --> Granite
    T3 --> Granite
    Granite -->|Structured JSON| Agent
    Agent -->|Formatted Report| User

    Frontend[🌐 Web Frontend\nHTML/CSS/JS] -->|REST API| Backend[🐍 FastAPI Backend\nPython]
    Backend -->|watsonx.ai API| Granite
    Backend -->|Orchestrate API| Agent

    style Agent fill:#4A90E2,stroke:#2E5C8A,color:#fff
    style Granite fill:#FF6B35,stroke:#CC4400,color:#fff
    style KB fill:#7C3AED,stroke:#5B21B6,color:#fff
    style F1 fill:#059669,stroke:#047857,color:#fff
    style T1 fill:#D97706,stroke:#B45309,color:#fff
    style T2 fill:#D97706,stroke:#B45309,color:#fff
    style T3 fill:#D97706,stroke:#B45309,color:#fff
    style Frontend fill:#1D4ED8,stroke:#1E40AF,color:#fff
    style Backend fill:#065F46,stroke:#064E3B,color:#fff
```

---

## Interview Prep Flow Diagram

```mermaid
flowchart TD
    Start([▶ START]) --> Input["📥 Receive Candidate Profile\nname, role, level, company type,\ndays until interview, context"]
    Input --> GenQ["🧠 Call IBM Granite\nGenerate 10 tailored questions\n4 technical + 3 behavioral +\n2 situational + 1 career goal"]
    GenQ --> ParseQ{"✅ JSON parsed\nsuccessfully?"}
    ParseQ -->|Yes| FormatQ["📝 Format Questions\nGrouped by category\nwith preparation tips"]
    ParseQ -->|No| FallbackQ["🔄 Use fallback\ndefault question set"]
    FallbackQ --> FormatQ
    FormatQ --> GenS["🧠 Call IBM Granite\nGenerate preparation strategy\ndaily plan + resources + tips"]
    GenS --> ParseS{"✅ JSON parsed\nsuccessfully?"}
    ParseS -->|Yes| FormatS["📅 Format Strategy\nDaily plan, key topics,\nresources, confidence tips"]
    ParseS -->|No| FallbackS["🔄 Use fallback\ndefault strategy"]
    FallbackS --> FormatS
    FormatS --> Combine["📄 Combine into\nFull Prep Report"]
    Combine --> End([⏹ END])

    style Start fill:#10B981,stroke:#059669,color:#fff
    style End fill:#EF4444,stroke:#DC2626,color:#fff
    style GenQ fill:#F59E0B,stroke:#D97706,color:#fff
    style GenS fill:#F59E0B,stroke:#D97706,color:#fff
    style ParseQ fill:#3B82F6,stroke:#2563EB,color:#fff
    style ParseS fill:#3B82F6,stroke:#2563EB,color:#fff
    style Combine fill:#8B5CF6,stroke:#7C3AED,color:#fff
```

---

## Answer Evaluation Flow

```mermaid
flowchart TD
    Start2([▶ START]) --> Q["❓ Candidate provides\ntheir practice answer"]
    Q --> Eval["🧠 IBM Granite Evaluates\nscore 1-10, strengths,\nimprovements, model answer"]
    Eval --> Score{"Score?"}
    Score -->|8-10| High["🌟 Excellent!\nHighlight strengths,\nminor refinements"]
    Score -->|5-7| Mid["📈 Good effort!\nSpecific improvements\n+ model answer"]
    Score -->|1-4| Low["🛠️ Needs work\nDetailed coaching\n+ full model answer"]
    High --> Return["📤 Return feedback\nto candidate"]
    Mid --> Return
    Low --> Return
    Return --> End2([⏹ END])

    style Start2 fill:#10B981,stroke:#059669,color:#fff
    style End2 fill:#EF4444,stroke:#DC2626,color:#fff
    style Eval fill:#F59E0B,stroke:#D97706,color:#fff
    style High fill:#10B981,stroke:#059669,color:#fff
    style Mid fill:#3B82F6,stroke:#2563EB,color:#fff
    style Low fill:#EF4444,stroke:#DC2626,color:#fff
```

---

## Project Structure

```
interview_trainer_agent/
├── __init__.py
├── main_flow.py                          # Programmatic testing script
├── import-all.sh                         # CLI import script
├── README.md                             # This file
├── knowledge_base/
│   └── interview_knowledge_base.md       # RAG knowledge base document
├── tools/
│   ├── __init__.py
│   ├── generate_interview_questions.py   # Tool: generate tailored questions
│   ├── evaluate_interview_answer.py      # Tool: score and evaluate answers
│   ├── build_preparation_strategy.py     # Tool: build daily prep plan
│   └── interview_prep_flow.py            # Flow: full combined prep report
├── agents/
│   ├── interview_trainer_agent.yaml      # Agent configuration
│   └── interview_knowledge_base.yaml     # Knowledge base spec
├── generated/
│   └── interview_prep_flow.json          # Compiled flow spec (auto-generated)
├── frontend/
│   └── index.html                        # Web UI for the interview trainer
└── backend/
    ├── app.py                            # FastAPI backend server
    └── requirements.txt                  # Python dependencies
```

---

## IBM Technology Used

| Component | IBM Technology |
|-----------|----------------|
| LLM | IBM Granite `ibm/granite-4-h-small` |
| Platform | IBM watsonx.ai (Cloud Lite) |
| Orchestration | IBM watsonx Orchestrate |
| Embeddings | `ibm/slate-125m-english-rtrvr-v2` |
| Vector Store | Built-in Milvus (managed) |

---

## Tools & Features

| Tool | Description |
|------|-------------|
| `generate_interview_questions` | Generates 10 tailored questions per profile |
| `evaluate_interview_answer` | Scores answers 1-10 with detailed feedback |
| `build_preparation_strategy` | Creates day-by-day preparation plans |
| `interview_prep_flow` | Full combined report: questions + strategy |

### Knowledge Base Content
- Role-specific technical questions (Software Engineering, Data Science, Product, Finance, Marketing)
- Behavioral question bank with STAR method guidance  
- Industry expectations by company type (FAANG, Startup, Enterprise)
- HR guidelines and common interview mistakes
- Salary negotiation and offer evaluation tips

---

## Usage

### Via watsonx Orchestrate Chat UI

```bash
# 1. Import everything
cd interview_trainer_agent
./import-all.sh

# 2. Start the chat
orchestrate chat start

# 3. Select interview_trainer_agent and start chatting
```

### Via Web Frontend

```bash
# Start the backend
cd interview_trainer_agent/backend
pip install -r requirements.txt
uvicorn app:app --reload --port 8000

# Open the frontend in a browser
open interview_trainer_agent/frontend/index.html
```

### Programmatic Testing (Flow)

```bash
export PYTHONPATH=/path/to/adk/src:/path/to/adk
cd /path/to/Interview_Trainer_Agent
python3 interview_trainer_agent/main_flow.py
```

---

## Example Interactions

**Generate Questions:**
> "I'm Alex, applying for a Senior Software Engineer role at a big tech company."

**Practice Mode:**
> "Ask me a behavioral question and I'll answer it."

**Get a Prep Plan:**
> "I have 14 days to prepare for a Data Scientist mid-level role at a startup."

**Full Report:**
> "Give me a complete interview prep report for a Product Manager entry-level role."

---

## Prerequisites

- IBM Cloud account with watsonx.ai access (Cloud Lite tier)
- watsonx Orchestrate environment set up
- `orchestrate` CLI installed and authenticated
- Python 3.10+
