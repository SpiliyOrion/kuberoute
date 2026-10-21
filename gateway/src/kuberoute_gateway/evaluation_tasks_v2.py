"""Sixty synthetic, uniquely keyed multiple-choice evaluation tasks.

The source-only notes make the logic and reading keys auditable. Neither notes
nor prompts are written to evaluation result artifacts.
"""

from dataclasses import dataclass

from kuberoute_gateway.evaluation_tasks import EvaluationTask

TASK_SET_V2_VERSION = "2"
CATEGORIES_V2 = (
    "arithmetic",
    "numerical_comparison",
    "logical_reasoning",
    "string_operations",
    "reading_comprehension",
)
CHOICES = "ABC"


@dataclass(frozen=True)
class TaskSpec:
    task_id: str
    question: str
    answer: str
    distractors: tuple[str, str]
    audit_note: str = ""


# Each category contains twelve different operations or scenarios. An answer
# rotates through A/B/C once every three rounds, including the first 15 tasks.
SPECS_V2: dict[str, tuple[TaskSpec, ...]] = {
    "arithmetic": (
        TaskSpec("v2_ar_add", "What is 17 + 8?", "25", ("24", "26")),
        TaskSpec("v2_ar_subtract", "What is 50 - 19?", "31", ("29", "32")),
        TaskSpec("v2_ar_multiply", "What is 6 times 7?", "42", ("36", "48")),
        TaskSpec("v2_ar_divide", "What is 81 divided by 9?", "9", ("8", "7")),
        TaskSpec(
            "v2_ar_remainder", "What is the remainder when 23 is divided by 5?", "3", ("2", "4")
        ),
        TaskSpec("v2_ar_parentheses", "What is (4 + 6) times 3?", "30", ("18", "24")),
        TaskSpec("v2_ar_average", "What is the mean of 8, 12, and 16?", "12", ("10", "14")),
        TaskSpec("v2_ar_percent", "What is 25 percent of 80?", "20", ("16", "25")),
        TaskSpec("v2_ar_fraction", "What is three quarters of 28?", "21", ("14", "24")),
        TaskSpec("v2_ar_negative", "What is -7 + 12?", "5", ("-5", "19")),
        TaskSpec("v2_ar_precedence", "What is 2 + 3 times 4?", "14", ("20", "24")),
        TaskSpec(
            "v2_ar_clock", "What time is 50 minutes after 14:35?", "15:25", ("15:15", "15:35")
        ),
    ),
    "numerical_comparison": (
        TaskSpec("v2_num_decimal", "Which positive decimal is greatest?", "0.705", ("0.7", "0.65")),
        TaskSpec("v2_num_negative", "Which number is smallest?", "-8", ("-3", "-1")),
        TaskSpec("v2_num_fraction", "Which fraction is closest to 1?", "4/5", ("3/4", "2/3")),
        TaskSpec("v2_num_length", "Which length is longest?", "1.6 m", ("150 cm", "1.4 m")),
        TaskSpec(
            "v2_num_time", "Which time is latest on the same day?", "10:05", ("09:45", "09:55")
        ),
        TaskSpec(
            "v2_num_absolute",
            "Which expression has the largest absolute value?",
            "|-9|",
            ("|7|", "|-8|"),
        ),
        TaskSpec(
            "v2_num_powers",
            "Which exponent expression has the greatest value?",
            "2 to the fifth power",
            ("3 to the third power", "5 squared"),
        ),
        TaskSpec("v2_num_closest", "Which number is closest to 2?", "1.85", ("1.7", "2.4")),
        TaskSpec(
            "v2_num_duration",
            "Which duration is shortest?",
            "45 seconds",
            ("1 minute", "0.8 minute"),
        ),
        TaskSpec("v2_num_equivalent", "Which fraction equals 0.95?", "19/20", ("9/10", "4/5")),
        TaskSpec(
            "v2_num_median",
            "Which number is the median of 408, 480, and 418?",
            "418",
            ("408", "480"),
        ),
        TaskSpec(
            "v2_num_gap",
            "Which pair has the smallest numerical difference?",
            "7 and 10",
            ("12 and 18", "20 and 24"),
        ),
    ),
    "logical_reasoning": (
        TaskSpec(
            "v2_log_transitive",
            "Mira is older than Nia. Nia is older than Oren. Which must be true?",
            "Mira is older than Oren",
            ("Oren is older than Mira", "Mira and Oren are the same age"),
            "Older-than is transitive.",
        ),
        TaskSpec(
            "v2_log_conjunction",
            "A gate opens only when both lamps are on. The blue lamp is off. Which must be true?",
            "The gate does not open",
            ("The gate opens", "Both lamps are on"),
            "A required lamp is off.",
        ),
        TaskSpec(
            "v2_log_subset",
            "Every tulip in a box is red. One item is a tulip from that box. Which must be true?",
            "The item is red",
            ("The item is blue", "Every red item is a tulip"),
            "The stated tulip belongs to the all-red set.",
        ),
        TaskSpec(
            "v2_log_order",
            "Ada arrived before Bo. Cy arrived after Bo. Who arrived first?",
            "Ada",
            ("Bo", "Cy"),
            "The order is Ada, Bo, Cy.",
        ),
        TaskSpec(
            "v2_log_even",
            "An integer is divisible by 4. Which must be true?",
            "It is even",
            ("It is odd", "It is divisible by 8"),
            "Every integer multiple of 4 is even.",
        ),
        TaskSpec(
            "v2_log_exclusive",
            "Of two switches, red and green, exactly one is on. Red is off. Which must be true?",
            "The green switch is on",
            ("The green switch is off", "Both switches are on"),
            "Exactly one is on, and red is off.",
        ),
        TaskSpec(
            "v2_log_schedule",
            "The bus leaves before the train. The ferry leaves after the train. Which leaves last?",
            "The ferry",
            ("The bus", "The train"),
            "The order is bus, train, ferry.",
        ),
        TaskSpec(
            "v2_log_implication",
            "If the card is square, it has four sides. This card is square. Which must be true?",
            "It has four sides",
            ("It has three sides", "It has five sides"),
            "Modus ponens applies to the stated square.",
        ),
        TaskSpec(
            "v2_log_remaining",
            "Exactly one of four cards is red. The first three are blue. Which is red?",
            "the fourth card",
            ("the first card", "the second card"),
            "With the first three blue, the only red card is fourth.",
        ),
        TaskSpec(
            "v2_log_position",
            "Three people stand in line: Lee before Mo before Pat. Who is in the middle?",
            "Mo",
            ("Lee", "Pat"),
            "The line is Lee, Mo, Pat.",
        ),
        TaskSpec(
            "v2_log_membership",
            "All keys on the hook are brass. The silver key is not brass. Which must be true?",
            "The silver key is not on the hook",
            ("The silver key is on the hook", "No brass keys are on the hook"),
            "A non-brass key cannot be among keys all known to be brass.",
        ),
        TaskSpec(
            "v2_log_opposite",
            "A light is either on or off, never both. It is not on. Which must be true?",
            "It is off",
            ("It is on", "It is both on and off"),
            "The two states are exhaustive and exclusive.",
        ),
    ),
    "string_operations": (
        TaskSpec("v2_str_upper", "How many uppercase letters are in 'aBcDEf'?", "3", ("2", "4")),
        TaskSpec("v2_str_reverse", "What is 'staple' reversed?", "elpats", ("elpat", "staple")),
        TaskSpec(
            "v2_str_sort",
            "What is the first letter after sorting the letters in 'pear' alphabetically?",
            "a",
            ("e", "p"),
        ),
        TaskSpec(
            "v2_str_spaces",
            "What is 'red fox' with its space removed?",
            "redfox",
            ("red fox", "red-fox"),
        ),
        TaskSpec(
            "v2_str_vowels", "How many vowels (a, e, i, o, u) are in 'cooperate'?", "5", ("4", "6")
        ),
        TaskSpec(
            "v2_str_suffix", "What are the last three letters of 'lantern'?", "ern", ("tern", "ant")
        ),
        TaskSpec(
            "v2_str_join",
            "Join 'north' and 'star' with one hyphen. What results?",
            "north-star",
            ("northstar", "star-north"),
        ),
        TaskSpec(
            "v2_str_index",
            "At what zero-based index does the first 'o' occur in 'coconut'?",
            "1",
            ("0", "3"),
        ),
        TaskSpec(
            "v2_str_replace",
            "Replace every 'a' in 'banana' with 'o'. What results?",
            "bonono",
            ("bonana", "banana"),
        ),
        TaskSpec("v2_str_distinct", "How many distinct letters are in 'level'?", "3", ("2", "5")),
        TaskSpec(
            "v2_str_lexical",
            "Which word comes first alphabetically?",
            "kitchen",
            ("kitten", "zebra"),
        ),
        TaskSpec(
            "v2_str_alternate",
            "From 'abcdefg', take every second letter starting with the second. What results?",
            "bdf",
            ("aceg", "bdfg"),
        ),
    ),
    "reading_comprehension": (
        TaskSpec(
            "v2_read_library",
            "A library opens at nine and closes at five. When does it open?",
            "nine",
            ("five", "noon"),
            "The passage says it opens at nine.",
        ),
        TaskSpec(
            "v2_read_garden",
            "Rae planted mint in the blue pot and basil in the white pot. Which pot has basil?",
            "white",
            ("blue", "green"),
            "Basil is explicitly in the white pot.",
        ),
        TaskSpec(
            "v2_read_trip",
            "The ferry stopped at East Pier before reaching West Pier. Which pier came first?",
            "East Pier",
            ("West Pier", "North Pier"),
            "The ferry stopped at East Pier before West Pier.",
        ),
        TaskSpec(
            "v2_read_delivery",
            "A parcel arrived on Tuesday, one day before the letter. What arrived Tuesday?",
            "the parcel",
            ("the letter", "both items"),
            "The parcel is explicitly the Tuesday arrival.",
        ),
        TaskSpec(
            "v2_read_recipe",
            "A recipe uses oats, milk, and pears but no eggs. Which ingredient is omitted?",
            "eggs",
            ("oats", "pears"),
            "The passage explicitly excludes eggs.",
        ),
        TaskSpec(
            "v2_read_roster",
            "Tess painted the sign. Uma fixed the door. Who painted the sign?",
            "Tess",
            ("Uma", "both"),
            "Tess is named as the painter.",
        ),
        TaskSpec(
            "v2_read_weather",
            "The morning was dry, but rain began after lunch. When did rain begin?",
            "after lunch",
            ("in the morning", "before sunrise"),
            "Rain began after lunch.",
        ),
        TaskSpec(
            "v2_read_ticket",
            "A red ticket admits Hall 2; a gold ticket admits Hall 3. Which hall accepts gold?",
            "Hall 3",
            ("Hall 2", "Hall 1"),
            "The gold ticket grants entry to Hall 3.",
        ),
        TaskSpec(
            "v2_read_box",
            "A box contains two maps and one compass. How many compasses are in it?",
            "one",
            ("two", "three"),
            "The passage states one compass.",
        ),
        TaskSpec(
            "v2_read_route",
            "The path crosses a bridge, then passes a clock tower. What follows the bridge?",
            "the clock tower",
            ("a station", "a fountain"),
            "The clock tower follows the bridge.",
        ),
        TaskSpec(
            "v2_read_color",
            "The small kite is orange; the large kite is purple. What color is the large kite?",
            "purple",
            ("orange", "yellow"),
            "The large kite is explicitly purple.",
        ),
        TaskSpec(
            "v2_read_meeting",
            "A notice lists workshop times: 14:00 to 15:00. When does it end?",
            "15:00",
            ("14:30", "16:00"),
            "The notice explicitly states an end time of 15:00.",
        ),
    ),
}


def _task(spec: TaskSpec, category: str, answer_index: int) -> EvaluationTask:
    options = list(spec.distractors)
    options.insert(answer_index, spec.answer)
    prompt = (
        f"{spec.question} A: {options[0]}. B: {options[1]}. C: {options[2]}. "
        "Reply with only the single answer letter."
    )
    return EvaluationTask(spec.task_id, category, prompt, CHOICES[answer_index])


TASKS_V2 = tuple(
    _task(SPECS_V2[category][round_index], category, (round_index + category_index) % 3)
    for round_index in range(12)
    for category_index, category in enumerate(CATEGORIES_V2)
)
