"""
main_flow.py — Programmatic testing script for the Interview Prep Flow.

Usage:
    export PYTHONPATH=/path/to/adk/src:/path/to/adk
    python3 main_flow.py
"""

import asyncio
from pathlib import Path

from interview_trainer_agent.tools.interview_prep_flow import build_interview_prep_flow


async def main():
    """Compile, deploy, and invoke the interview preparation flow for testing."""

    print("=" * 60)
    print("  Interview Trainer Agent — Flow Test")
    print("=" * 60)

    # Compile and deploy the flow
    print("\n⚙️  Compiling and deploying flow...")
    flow_def = await build_interview_prep_flow().compile_deploy()

    # Save the generated flow spec
    generated_folder = Path(__file__).resolve().parent / "generated"
    generated_folder.mkdir(exist_ok=True)
    flow_def.dump_spec(str(generated_folder / "interview_prep_flow.json"))
    print(f"📄 Flow spec saved to: {generated_folder}/interview_prep_flow.json")

    # Test invocation
    print("\n🚀 Invoking flow with test candidate profile...")
    result = await flow_def.invoke(
        {
            "profile_name": "Alex Johnson",
            "job_role": "Software Engineer",
            "experience_level": "mid",
            "target_company_type": "big_tech",
            "days_until_interview": 10,
            "additional_context": "5 years Python backend experience, familiar with AWS and microservices",
        },
        debug=True,
    )

    print("\n" + "=" * 60)
    print("  FLOW OUTPUT")
    print("=" * 60)
    if result:
        print(result)

    print("\n✅ Flow test completed successfully!")


if __name__ == "__main__":
    asyncio.run(main())
