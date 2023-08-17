from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree

from . import ROOT_DIR

LANGUAGES_SO = str(Path(ROOT_DIR, "build", "languages.so").resolve())
TREE_SITTER_TS = str(Path(ROOT_DIR, "tree-sitter-typescript", "typescript").resolve())

Language.build_library(LANGUAGES_SO, [TREE_SITTER_TS])

LANGUAGE = Language(LANGUAGES_SO, "typescript")
PARSER = Parser()
PARSER.set_language(LANGUAGE)

def parse(s: str) -> Tree:
    """Parses the given string into a tree."""
    return PARSER.parse(s.encode("utf-8"))

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
    captures = run_query(content,
        """
        (arrow_function
            parameter: (identifier) @param
            body: (_))
        """)
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
    captures = run_query(content,
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
        """)

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

def delete_nodes(content: str, nodes: list[Node]) -> str:
    """
    Given a content string and a list of AST nodes, return a new string with
    those nodes deleted.
    """
    # Need to operate on byte string, not characters
    content_bytes = content.encode("utf-8")

    # Flatten the pairs of indices into a list.
    # We add None to the beginning and end of the list of indices, so we can do
    # array slicing.
    # e.g. [(s1, e1), (s2, e2)]
    #   -> [None, s1, e1, s2, e2, None]
    pairs = [(n.start_byte, n.end_byte) for n in nodes]
    indices = [None] + [i for p in pairs for i in p] + [None]

    # We zip the list with itself (offset by 1), moving by 2 elements each time,
    # e.g. [None, s1, e1, s2, e2, None]
    #   -> [(None, s1), (e1, s2), (e2, None)]
    chunks = []
    for s, e in zip(indices[::2], indices[1::2]):
        chunks.append(content_bytes[s:e].decode("utf-8-sig"))
    new_content = "".join(chunks)

    return new_content

def is_child_type_annotation(start_node: Node) -> bool:
    """Checks if any of the parent nodes is an annotation node."""
    node = start_node.parent
    while node is not None:
        if (node.type == "type_annotation" or
                node.type == "opting_type_annotation" or
                node.type == "omitting_type_annotation"):
            return True
        node = node.parent
    return False

def delete_type_annotations(content: str) -> str:
    """Deletes type annotations from the given string."""
    captures = run_query(content,
        """
        [
            (type_annotation) @annotation
            (opting_type_annotation) @annotation
            (omitting_type_annotation) @annotation
        ]
        """)
    nodes = [c[0] for c in captures
             if not is_child_type_annotation(c[0])]

    return delete_nodes(content, nodes)

def is_child_of_export(start_node: Node) -> bool:
    """Checks if the parent node is an export statement."""
    if start_node.parent:
        return start_node.parent.type == "export_statement"
    return False

def delete_type_definitions(content: str) -> str:
    """Deletes type definitions from the given string."""
    captures = run_query(content,
        """
        [
            (interface_declaration) @type
            (class_declaration) @type
            (abstract_class_declaration) @type
            (type_alias_declaration) @type
            (export_statement
                declaration: (interface_declaration)) @type
            (export_statement
                declaration: (class_declaration)) @type
            (export_statement
                declaration: (abstract_class_declaration)) @type
            (export_statement
                declaration: (type_alias_declaration)) @type
        ]
        """)
    nodes = [c[0] for c in captures if not is_child_of_export(c[0])]
    return delete_nodes(content, nodes)

def delete_types(content: str) -> str:
    """Deletes type annotations and type definitions from the given string."""
    content = delete_type_definitions(content)
    content = delete_type_annotations(content)
    return content

def delete_comments(content: str) -> str:
    """Deletes comments from the given string."""
    captures = run_query(content, "(comment) @comment")
    nodes = [c[0] for c in captures]
    return delete_nodes(content, nodes)

def is_empty(content: str) -> bool:
    """Returns true if the content contains only whitespace and comments."""
    return delete_comments(content).strip() == ""
