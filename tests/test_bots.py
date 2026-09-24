from pathlib import Path

from evalharness.bots import REFUSAL, GuardedBot
from evalharness.retriever import Retriever

KB = Path(__file__).resolve().parents[1] / "data/knowledge_base"


def bot():
    return GuardedBot(Retriever(KB))


def test_answers_known_fact():
    assert "acetylcysteine" in bot().ask("What is the antidote for paracetamol overdose?").answer.lower()


def test_refuses_unknown_drug():
    assert bot().ask("What is the maximum dose of Zentrolex?").answer == REFUSAL


def test_refuses_off_topic():
    assert bot().ask("What is the capital of France?").answer == REFUSAL
