from typing import Any, Callable, Optional
import argparse
import datasets
import functools
import random

from util import transform
import util

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

def default(element: dict[str, Any]) -> Optional[str]:
    return element["content"]

def get1(element: dict[str, Any]) -> Optional[str]:
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

def get2(element: dict[str, Any]) -> Optional[str]:
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
        return commit_format(no_anns, "Add type annotations", original)

def get3(element: dict[str, Any]) -> Optional[str]:
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
        return commit_format(no_types, "Add type annotations", no_defs)
    else:
        return commit_format(
            no_defs, "Add type aliases and interfaces", original
        )

def get4(element: dict[str, Any]) -> Optional[str]:
    """
    This approach relies on multiple steps to infer type definitions. First, the
    model inserts type annotations. Next, in a loop, the model generates type
    definitions one at a time, based on the type annotations that refer to
    undefined types.

    This requires two different training formats. The first format adds type
    annotations. The second format randomly removes some number of type
    definitions, and then adds one back in the "after" code.

    Format 1:
        <commit_before>{code without types}
        <commit_msg>Add type annotations
        <commit_after>{code with type annotations but no type definitions}

    Format 2:
        <commit_before>{code with type annotations and some type definitions}
        <commit_msg>Add a type alias or interface for T
        <commit_after>{code with type annotations and some type definitions,
        plus a definition for T}
    """
    original = element["content"].strip()

    # If there are no type definitions, use the first format
    typedef_nodes = transform.extract_type_definition_nodes(original)
    if len(typedef_nodes) == 0:
        no_defs = transform.delete_type_definitions(original).strip()

        # Don't forget to delete type assertions!
        no_types = transform.delete_type_annotations(no_defs)
        no_types = transform.delete_type_assertions(no_types).strip()

        return commit_format(no_types, "Add type annotations", no_defs)
    else:
        # For each node, flip a coin to determine if we delete or keep it
        # Make sure we delete at least one node
        to_delete = []
        while len(to_delete) == 0:
            to_delete = [n for n in typedef_nodes if random.randint(0, 1) == 0]

        # Delete types from the code, to get the commit before
        before_code = transform.delete_nodes(original, to_delete)

        # Select the node at a random index to keep, and remove it from the list
        # The remaining nodes are ones we want to delete
        index = random.randrange(0, len(to_delete))
        keep_node = to_delete.pop(index)

        # Delete types from the code, to get the commit after
        after_code = transform.delete_nodes(original, to_delete)

        # Get the name of the type we're keeping, so we can add it to the prompt
        type_to_add = transform.get_name_from_type_definition(keep_node)

        return commit_format(
            before_code,
            f"Add a type alias or interface for {type_to_add}",
            after_code
        )

APPROACHES = {
    "get1": get1,
    "get2": get2,
    "get3": get3,
    "get4": get4
}

def _transform_example(
    example: dict[str, Any],
    approach: Callable[[dict[str, Any]], str]
) -> dict[str, Any]:
    example["content"] = approach(example)
    return example

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=util.cpu_count())
    parser.add_argument("--approach", choices=APPROACHES.keys(), required=True)
    parser.add_argument("--output", type=str, required=True)

    return parser.parse_args()

def main():
    """
    Use this script to preprocess a dataset and save it to disk.
    """
    args = get_args()
    random.seed(args.seed)
    datasets.disable_caching()

    dataset = util.load_dataset(
        "nuprl/ts-training",
        split="train",
        revision="v1.1p1",
        num_proc=args.workers
    )

    new_dataset = dataset.map(
        functools.partial(_transform_example, approach=APPROACHES[args.approach]),
        num_proc=args.workers
    )
    util.save_dataset(new_dataset, args.output, args.workers)

if __name__ == "__main__":
    main()
