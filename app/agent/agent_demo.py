"""Module 3 demo: the agent loop in action.

Run:  python -m app.agent.agent_demo
"""
from langchain_core.messages import HumanMessage
from app.agent.graph import build_graph

from app.core.metrics import track
from app.core.pricing import cost_usd
from app.core.config import get_settings

def run_agent(user_input: str) -> str:
    graph = build_graph()
    with track("agent_run") as m:
        result = graph.invoke({"messages": [HumanMessage(content=user_input)]})
        for msg in result["messages"]:
            usage = getattr(msg, "usage_metadata", None)
            if usage:
                m["input_tokens"] += usage.get("input_tokens", 0)
                m["output_tokens"] += usage.get("output_tokens", 0)
        m["cost_usd"] = cost_usd(
            get_settings().llm_model, m["input_tokens"], m["output_tokens"]
        )
    return result["messages"][-1].content


def main() -> None:
    # Test 1: pure reasoning, no tools needed
    print("=== TEST 1: no tools ===")
    print(run_agent("What is the capital of Greece?"))

    # Test 2: forces the agent to use write_file then read_file
    print("\n=== TEST 2: file tools ===")
    print(run_agent(
        "Write a file at /tmp/jobs.txt with three tips for job hunting in tech. "
        "Then read it back and confirm what you wrote."
    ))

    # Test 3: shell tool
    print("\n=== TEST 3: shell tool ===")
    print(run_agent("How many lines are in /tmp/jobs.txt?"))


if __name__ == "__main__":
    main()