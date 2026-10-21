"""Versioned, harmless synthetic tasks for small routing evaluations."""

from dataclasses import dataclass

TASK_SET_VERSION = "1"


@dataclass(frozen=True)
class EvaluationTask:
    task_id: str
    category: str
    prompt: str
    expected_choice: str | None = None


WARMUP_TASK = EvaluationTask("warmup_ack", "warmup", "Reply with only the word OK.")

TASKS = (
    EvaluationTask(
        "add_2_3",
        "arithmetic",
        "What is 2 + 3? A: 4. B: 5. C: 6. Reply with the single correct letter only.",
        "B",
    ),
    EvaluationTask(
        "multiply_3_4",
        "arithmetic",
        "What is 3 times 4? A: 9. B: 10. C: 12. Reply with the single correct letter only.",
        "C",
    ),
    EvaluationTask(
        "compare_7_4",
        "comparison",
        "Which number is larger? A: 7. B: 4. Reply with the single correct letter only.",
        "A",
    ),
    EvaluationTask(
        "even_8",
        "classification",
        "Is 8 even? A: yes. B: no. Reply with the single correct letter only.",
        "A",
    ),
    EvaluationTask(
        "describe_blue_cube",
        "open_ended",
        "Write one short sentence describing a fictional blue cube.",
    ),
)


def task_for_sequence(sequence: int) -> EvaluationTask:
    """Repeat the same ordered task cycle for every strategy."""
    return TASKS[sequence % len(TASKS)]
