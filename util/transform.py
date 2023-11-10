from functools import lru_cache
from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree
from typing import Optional

from . import ROOT_DIR

LANGUAGES_SO = str(Path(ROOT_DIR, "build", "languages.so").resolve())
TREE_SITTER_TS = str(Path(ROOT_DIR, "tree-sitter-typescript", "typescript").resolve())

Language.build_library(LANGUAGES_SO, [TREE_SITTER_TS])

LANGUAGE = Language(LANGUAGES_SO, "typescript")
PARSER = Parser()
PARSER.set_language(LANGUAGE)


@lru_cache(maxsize=32)
def parse(s: str) -> Tree:
    """
    Parses the given string into a tree. Uses lru_cache so we don't need to
    repeatedly parse the same string.
    """
    return PARSER.parse(s.encode("utf-8"))


def is_valid_syntax(s: str) -> bool:
    """Returns true if the given string is syntactically valid TypeScript."""
    root = parse(s).root_node
    return not root.has_error


def node_to_str(node: Node) -> str:
    """Returns the string represented by the given tree node."""
    # utf-8-sig skips the BOM (if it exists) when decoding
    return node.text.decode("utf-8-sig")


def run_query(content: str, query_str: str) -> list[tuple[Node, str]]:
    """
    Runs a query on the given program fragment. Returns a list of captures, i.e.
    a list of tuples where the first item is a Node and the second is the name
    of the capture.
    """
    tree = parse(content)
    query = LANGUAGE.query(query_str)
    return query.captures(tree.root_node)


def parenthesize_arrow_params(content: str) -> str:
    """
    Ensures arrow functions have their parameters wrapped with parentheses.
    This allows type annotation prediction to be uniform (since type
    annotations for single-parameter arrow functions require the
    parentheses).

    Example:  x => x      becomes      (x) => x
    """
    captures = run_query(
        content,
        """
        (arrow_function
            parameter: (identifier) @param
            body: (_))
        """,
    )
    pairs = [(c[0].start_byte, c[0].end_byte) for c in captures]

    # Given pairs of start and end bytes for the unparenthesized parameters,
    # iterate over the byte array in reverse order (so we don't offset the
    # indices) and insert ')' and '(' at the appropriate locations.
    content_bytes = bytearray(content.encode("utf-8"))
    for s, e in reversed(pairs):
        content_bytes[e:e] = bytearray(")".encode("utf-8"))
        content_bytes[s:s] = bytearray("(".encode("utf-8"))

    return content_bytes.decode("utf-8-sig")


def split_at_annotation_locations(content: str) -> list[str]:
    """
    Given the TypeScript program as a string, split at the type annotation
    locations and return a list of strings, where each item is the substring
    between annotation locations.

    Type annotation locations are immediately after:
        - function parameters         function f(a: TYPE)
        - formal parameter lists      function g(x): TYPE
        - public field definitions    class { x: TYPE }
        - variable declarators        let x: TYPE
    """
    captures = run_query(
        content,
        """
        [
            (required_parameter
                pattern: (_) @ann)
            (optional_parameter
                pattern: (_)) @ann
            (formal_parameters) @ann
            (public_field_definition
                name: (_) @ann)
            (variable_declarator
                name: (_) @ann)
        ]
        """,
    )

    # Need to operate on byte string, not characters
    content_bytes = content.encode("utf-8")

    # The end byte is where we want to insert type annotations.
    # We add None to the beginning and end of the list of indices, so we can do
    # array slicing, e.g. a[None:i] === a[:i]
    indices = [None] + sorted([c[0].end_byte for c in captures]) + [None]

    # Zip the list with itself, offset by 1.
    # e.g. [None, i1, i2, None]
    #   -> [(None, i1), (i1, i2), (i2, None)]
    # This gives us the chunks:
    #      c[:i1], c[i1:i2], c[i2:]
    # And we want to insert type annotations between the chunks.
    chunks = []
    for s, e in zip(indices, indices[1:]):
        chunks.append(content_bytes[s:e].decode("utf-8-sig"))

    return chunks


def slice_input(content: str, slice_length: int) -> list[str]:
    """
    Splits the input into slices of at most slice_length, breaking at newlines
    whenever possible.
    """
    slices = []
    curr_slice, curr_len = "", 0
    for line in content.splitlines(keepends=True):
        line_len = len(line)
        if curr_len + line_len > slice_length:
            slices.append(curr_slice)
            curr_slice, curr_len = "", 0
        curr_slice += line
        curr_len += line_len
    slices.append(curr_slice)
    return slices


def prefix_ending_with_newline(s: str, max_length: int) -> str:
    """
    Produces a prefix of s that is at most max_length, but does not split a
    line.
    """
    return s[:max_length].rsplit("\n", 1)[0]


def suffix_starting_with_newline(s: str, max_length: int) -> str:
    """
    Produces a suffix of s that is at most max_length, but does not split a
    line.
    """
    return s[-max_length:].split("\n", 1)[-1]


def clip_text(s1: str, s2: str, max_length: int) -> tuple[str, str]:
    """
    Clips s1 and s2 so that each string is at most half of
    max_length (which is measured in characters).
    """
    if len(s1) < max_length // 2:
        # s1 is short enough, so get the longest prefix of s2
        s2 = prefix_ending_with_newline(s2, max_length - len(s1))
    elif len(s2) < max_length // 2:
        # s2 is short enough, so get the longest suffix of s1
        s1 = suffix_starting_with_newline(s1, max_length - len(s2))
    else:
        # Both strings are too long, so clip both of them
        s1 = suffix_starting_with_newline(s1, max_length // 2)
        s2 = prefix_ending_with_newline(s2, max_length // 2)
    return s1, s2


def delete_between_indices(content: str, indices: list[Optional[int]]) -> str:
    """Helper for deleting text between indices of a string."""
    # Need to operate on byte string, not characters
    content_bytes = content.encode("utf-8")

    # We zip the list with itself (offset by 1), moving by 2 elements each time,
    # e.g. [None, s1, e1, s2, e2, None]
    #   -> [(None, s1), (e1, s2), (e2, None)]
    chunks = []
    for s, e in zip(indices[::2], indices[1::2]):
        chunks.append(content_bytes[s:e].decode("utf-8-sig"))
    new_content = "".join(chunks)

    return new_content


def delete_nodes(content: str, nodes: list[Node]) -> str:
    """
    Given a content string and a list of AST nodes, return a new string with
    those nodes deleted.
    """
    # Flatten the pairs of indices into a list.
    # We add None to the beginning and end of the list of indices, so we can do
    # array slicing.
    # e.g. [(s1, e1), (s2, e2)]
    #   -> [None, s1, e1, s2, e2, None]
    pairs = [(n.start_byte, n.end_byte) for n in nodes]
    indices = [None] + [i for p in pairs for i in p] + [None]

    return delete_between_indices(content, indices)


def is_child_type_annotation(start_node: Node) -> bool:
    """Checks if any of the parent nodes is an annotation node."""
    node = start_node.parent
    while node is not None:
        if (
            node.type == "type_annotation"
            or node.type == "opting_type_annotation"
            or node.type == "omitting_type_annotation"
        ):
            return True
        node = node.parent
    return False


def extract_type_annotation_nodes(content: str) -> list[Node]:
    """
    Returns a list of nodes, representing type annotations from the given string.
    """
    captures = run_query(
        content,
        """
        [
            (type_annotation) @annotation
            (opting_type_annotation) @annotation
            (omitting_type_annotation) @annotation
        ]
        """,
    )
    nodes = [c[0] for c in captures if not is_child_type_annotation(c[0])]
    return nodes


def delete_type_annotations(content: str) -> str:
    """Deletes type annotations from the given string."""
    nodes = extract_type_annotation_nodes(content)
    return delete_nodes(content, nodes)


def is_child_of_export(start_node: Node) -> bool:
    """Checks if the parent node is an export statement."""
    if start_node.parent:
        return start_node.parent.type == "export_statement"
    return False


def extract_type_definition_nodes(
    content: str, include_classes: bool = False, delete_comments: bool = False
) -> list[Node]:
    """
    Returns a list of nodes, representing (non-class) type definitions from the
    given string.
    """
    classes_query = """
        (class_declaration) @type
        (export_statement
            declaration: (class_declaration)) @type
    """
    type_definitions_query = f"""
    [
        (interface_declaration) @type
        (type_alias_declaration) @type
        (export_statement
            declaration: (interface_declaration)) @type
        (export_statement
            declaration: (type_alias_declaration)) @type
        {classes_query if include_classes else ""}
    ]
    """
    captures = run_query(content, type_definitions_query)
    nodes = [c[0] for c in captures if not is_child_of_export(c[0])]

    # If deleting comments that directly precede type definition nodes,
    # check the previous (named) sibling of each type definition node
    if delete_comments:
        res = []
        for n in nodes:
            prev_sibling = n.prev_named_sibling
            if prev_sibling and prev_sibling.type == "comment":
                res.append(prev_sibling)
            res.append(n)
        return res

    return nodes


def delete_type_definitions(content: str, delete_comments: bool = False) -> str:
    """Deletes (non-class) type definitions from the given string."""
    nodes = extract_type_definition_nodes(content, delete_comments=delete_comments)
    return delete_nodes(content, nodes)


def get_type_name(node: Node) -> str:
    """
    Given a type annotation or definition node, return the name of the type.
    """
    query = LANGUAGE.query(
        """
        [
          (predefined_type) @name
          (type_identifier) @name
        ]
        """
    )
    captures = query.captures(node)
    return node_to_str(captures[0][0]).strip()


def get_type_identifier_name(node: Node) -> Optional[str]:
    """
    Given a type annotation or definition node, return the name of type type
    if it is a user-defined type. Otherwise, return None.
    """
    query = LANGUAGE.query("(type_identifier) @name")
    captures = query.captures(node)
    if captures:
        return node_to_str(captures[0][0]).strip()
    else:
        return None


def is_child_type_assertion(start_node: Node) -> bool:
    """Checks if any of the parent nodes is a type assertion node."""
    node = start_node.parent
    while node is not None:
        if node.type == "as_expression":
            return True
        node = node.parent
    return False


def delete_type_assertions(content: str) -> str:
    """Recursively deletes type assertions from the given string."""
    captures = run_query(content, "(as_expression) @as_expression")

    nodes = [c[0] for c in captures if not is_child_type_assertion(c[0])]
    if not nodes:
        return content

    # Make a list of indices for deletion, to be used by delete_between_indices
    # i.e. the format needs to be [None, s1, e1, ..., s2, e2, None]
    # where `s` and `e` are `start` and `end` indices
    indices: list[Optional[int]] = [None]
    for n in nodes:
        # `42 as number` is an as_expression, where the first child is `42` and
        # the second child is `number`. We want to delete `as number`, i.e. from
        # the end of the first child to the end of the as_expression.
        indices.append(n.children[0].end_byte)
        indices.append(n.end_byte)
    indices.append(None)

    content = delete_between_indices(content, indices)

    # Recursive call to handle nested type assertion expressions
    return delete_type_assertions(content)


def delete_types(content: str, delete_comments: bool = False) -> str:
    """Deletes type annotations and type definitions from the given string."""
    content = delete_type_definitions(content, delete_comments)
    content = delete_type_annotations(content)
    content = delete_type_assertions(content)
    return content


def delete_comments(content: str) -> str:
    """Deletes comments from the given string."""
    captures = run_query(content, "(comment) @comment")
    nodes = [c[0] for c in captures]
    return delete_nodes(content, nodes)


def is_empty(content: str) -> bool:
    """Returns true if the content contains only whitespace and comments."""
    return delete_comments(content).strip() == ""


def get_undefined_type_names(content: str) -> list[str]:
    """
    Return a list of undefined type names, for the given string.
    """
    type_anns = {
        get_type_identifier_name(n) for n in extract_type_annotation_nodes(content)
    }
    type_defs = {
        get_type_identifier_name(n)
        for n in extract_type_definition_nodes(content, include_classes=True)
    }
    result = [t for t in (type_anns - type_defs) if t]
    return result
