from typing import Optional
import random

from util import transform

"""
This file contains several implementations of transforming a training example
into different formats for fine tuning a model to generate (TypeScript) type
definitions.
"""

COMMIT_BEFORE = "<commit_before>"
COMMIT_MSG = "<commit_msg>"
COMMIT_AFTER = "<commit_after>"

def commit_format(before: str, message: str, after: str) -> Optional[str]:
    """
    Transform into git commit training format.

    Returns None if `after` or `before` are empty, or they are identical to
    each other.
    """
    if transform.is_empty(before) or transform.is_empty(after) or before == after:
        return None

    return (
        f"{COMMIT_BEFORE}{before}"
        f"{COMMIT_MSG}{message}"
        f"{COMMIT_AFTER}{after}"
    )

def default(element: dict) -> Optional[str]:
    return element["content"]

def get1(element: dict) -> Optional[str]:
    """
    We transform an input example (e.g. the contents of a TypeScript file) into
    the StarCoder git commit format, e.g.

        <commit_before>{code without types}
        <commit_msg>Add type annotations and interfaces
        <commit_after>{original code}

    Returns None if the processed element is invalid; i.e. removing all the types
    removes the entire content, or there were no types to remove.
    """
    original = element["content"].strip()
    without_types = transform.delete_types(original).strip()

    return commit_format(
        without_types, "Add type annotations and interfaces", original
    )

def get2(element: dict) -> Optional[str]:
    """
    This approach relies on two steps to infer type definitions: first the model
    generates type definitions, then it inserts type annotations.

    This requires two different formats. For training, we flip a coin to
    determine which format to use.

    Format 1:
        <commit_before>{code without types}
        <commit_msg>Add type aliases and interfaces
        <commit_after>{code with type definitions but no type annotations}

    Format 2:
        <commit_before>{code with type definitions but no type annotations}
        <commit_msg>Add type annotations
        <commit_after>{original code}
    """
    original = element["content"].strip()

    # Don't forget to delete type assertions!
    no_anns = transform.delete_type_annotations(original)
    no_anns = transform.delete_type_assertions(no_anns).strip()

    no_types = transform.delete_type_definitions(no_anns).strip()

    # Flip a coin to determine which training format to use
    if random.randint(0, 1) == 0:
        return commit_format(
            no_types, "Add type aliases and interfaces", no_anns
        )
    else:
        return commit_format(
            no_anns, "Add type annotations", original
        )

def get3(element: dict) -> Optional[str]:
    """
    This approach relies on two steps to infer type definitions: first the model
    inserts type annotations, then it generates type definitions.

    This requires two different formats. For training, we flip a coin to
    determine which format to use.

    Format 1:
        <commit_before>{code without types}
        <commit_msg>Add type annotations
        <commit_after>{code with type annotations but no type definitions}

    Format 2:
        <commit_before>{code with type annotations but no type definitions}
        <commit_msg>Add type aliases and interfaces
        <commit_after>{original code}
    """
    original = element["content"].strip()

    no_defs = transform.delete_type_definitions(original).strip()

    # Don't forget to delete type assertions!
    no_types = transform.delete_type_annotations(no_defs)
    no_types = transform.delete_type_assertions(no_types).strip()

    # Flip a coin to determine which training format to use
    if random.randint(0, 1) == 0:
        return commit_format(
            no_types, "Add type annotations", no_defs
        )
    else:
        return commit_format(
            no_defs, "Add type aliases and interfaces", original
        )
