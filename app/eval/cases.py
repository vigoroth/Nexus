"""Eval test cases. Each case is a sequence of turns sent to the agent in one
conversation (one thread), with expected keywords checked against the FINAL answer.
"""
from dataclasses import dataclass, field


@dataclass
class EvalCase:
    name: str                          # short id for the case
    turns: list[str]                   # messages sent in order, same conversation
    expect_any: list[str] = field(default_factory=list)   # answer must contain >=1 of these
    expect_all: list[str] = field(default_factory=list)   # answer must contain ALL of these


@dataclass
class CrossConvCase:
    name: str
    store_turns: list[str]      # sent in conversation A (must trigger long-term save)
    recall_turn: str            # sent in conversation B (fresh thread)
    expect_any: list[str] = field(default_factory=list)
    expect_all: list[str] = field(default_factory=list)


CROSS_CONV_CASES = [
    CrossConvCase(
        name="x_recall_name",
        store_turns=["Please remember that my name is Vigoroth. Save this to memory."],
        recall_turn="What is my name? Check your memory.",
        expect_any=["Vigoroth"],
    ),
    CrossConvCase(
        name="x_recall_goal",
        store_turns=["Remember for later: I'm targeting ML engineering roles. Save it."],
        recall_turn="What roles am I targeting? Look it up in memory.",
        expect_any=["ML", "machine learning"],
    ),
    CrossConvCase(
        name="x_recall_db_region",
        store_turns=["Important to remember: my production database is in us-east-1. Save to memory."],
        recall_turn="Which region is my production database in? Check memory.",
        expect_any=["us-east-1", "east"],
    ),
]





# memory-recall cases: tell the agent something, then ask about it later
MEMORY_CASES = [
    EvalCase(
        name="recall_name",
        turns=[
            "My name is Vigoroth.",
            "What is my name?",
        ],
        expect_any=["Vigoroth"],
    ),
    EvalCase(
        name="recall_goal",
        turns=[
            "I'm targeting ML engineering roles in my job search.",
            "What kind of roles am I looking for?",
        ],
        expect_any=["ML", "machine learning"],
    ),
    EvalCase(
        name="recall_fact_after_distractor",
        turns=[
            "My production database is on us-east-1.",
            "What's the weather like generally in spring?",
            "Which region is my production database in?",
        ],
        expect_any=["us-east-1", "east"],
    ),
    EvalCase(
        name="recall_preference",
        turns=[
            "I prefer Python over JavaScript for backend work.",
            "What language do I prefer for backend?",
        ],
        expect_any=["Python"],
    ),
    EvalCase(
        name="recall_multi_fact",
        turns=[
            "I have two cats named Pixel and Vector.",
            "What are my cats' names?",
        ],
        expect_all=["Pixel", "Vector"],
    ),
]