#!/usr/bin/env bash
# =============================================================================
# import-all.sh — Import the Interview Trainer Agent into watsonx Orchestrate
# =============================================================================
# Usage: ./import-all.sh
# Prerequisites: `orchestrate` CLI installed and authenticated
# =============================================================================

set -euo pipefail

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

echo ""
echo "============================================================"
echo "  Interview Trainer Agent — watsonx Orchestrate Import"
echo "============================================================"
echo ""

# ── Step 1: Import Knowledge Base ─────────────────────────────────────────────
echo "📚 Importing Knowledge Base..."
orchestrate knowledge-bases import -f "${SCRIPT_DIR}/agents/interview_knowledge_base.yaml" \
  -p "${SCRIPT_DIR}/knowledge_base"
echo "✅ Knowledge base imported."
echo ""

# ── Step 2: Import Python Tools ───────────────────────────────────────────────
echo "🔧 Importing Python tools..."

for tool_file in \
    generate_interview_questions.py \
    evaluate_interview_answer.py \
    build_preparation_strategy.py; do
  echo "  → Importing ${tool_file}..."
  orchestrate tools import -k python -f "${SCRIPT_DIR}/tools/${tool_file}"
done

echo "✅ Python tools imported."
echo ""

# ── Step 3: Import Flow Tool ──────────────────────────────────────────────────
echo "🔀 Importing Flow tool..."
orchestrate tools import -k flow -f "${SCRIPT_DIR}/tools/interview_prep_flow.py"
echo "✅ Flow tool imported."
echo ""

# ── Step 4: Import Agent ──────────────────────────────────────────────────────
echo "🤖 Importing Interview Trainer Agent..."
orchestrate agents import -f "${SCRIPT_DIR}/agents/interview_trainer_agent.yaml"
echo "✅ Agent imported."
echo ""

echo "============================================================"
echo "  🎉 Import complete!"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Verify the agent: orchestrate agents list"
echo "  2. Start a chat session: orchestrate chat start"
echo "  3. Select 'interview_trainer_agent' to begin"
echo ""
