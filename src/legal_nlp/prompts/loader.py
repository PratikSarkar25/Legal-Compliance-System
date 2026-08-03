"""
Utility functions for loading official CUAD prompts.

The prompts are stored separately from the codebase so that they can
be updated independently of the clause detection logic.
"""

import json
from pathlib import Path
from typing import Any, Dict, List


PROMPTS_DIR = Path(__file__).parent
CUAD_QUESTIONS_FILE = PROMPTS_DIR / "cuad_questions.json"


def load_cuad_questions() -> List[Dict[str, Any]]:
    """
    Load the official CUAD Question Answering prompts.

    Returns
    -------
    List[Dict[str, Any]]
        Example:

        [
            {
                "id": 1,
                "clause_type": "document_name",
                "display_name": "Document Name",
                "prompt": "Highlight the parts (if any)..."
            },
            ...
        ]

    Raises
    ------
    FileNotFoundError
        If the JSON file cannot be found.

    ValueError
        If the JSON file is malformed.
    """

    if not CUAD_QUESTIONS_FILE.exists():
        raise FileNotFoundError(
            f"CUAD questions file not found: {CUAD_QUESTIONS_FILE}"
        )

    try:
        with open(CUAD_QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions = json.load(f)

    except json.JSONDecodeError as e:
        raise ValueError(
            f"Invalid JSON in {CUAD_QUESTIONS_FILE}"
        ) from e

    if not isinstance(questions, list):
        raise ValueError(
            "CUAD questions JSON must contain a list."
        )

    return questions


def get_question_by_clause_type(
    clause_type: str,
) -> Dict[str, Any]:
    """
    Retrieve a CUAD question by its clause type.

    Parameters
    ----------
    clause_type : str
        Internal clause identifier
        (e.g. 'governing_law').

    Returns
    -------
    Dict[str, Any]
        Matching CUAD question.

    Raises
    ------
    KeyError
        If the clause type does not exist.
    """

    questions = load_cuad_questions()

    for question in questions:
        if question["clause_type"] == clause_type:
            return question

    raise KeyError(
        f"No CUAD question found for clause type '{clause_type}'."
    )


def get_all_clause_types() -> List[str]:
    """
    Return all supported CUAD clause identifiers.

    Returns
    -------
    List[str]
    """

    return [
        question["clause_type"]
        for question in load_cuad_questions()
    ]