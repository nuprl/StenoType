from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree
from typing import Optional

LANGUAGES_SO = f"{Path(__file__).parent}/build/languages.so"
TREE_SITTER_TS = f"{Path(__file__).parent}/tree-sitter-typescript/typescript"

Language.build_library(LANGUAGES_SO, [TREE_SITTER_TS])

class TypeInference:
    """
    This class does type inference (specifically, type annotation prediction)
    for TypeScript. It parses an unannotated TypeScript file with tree_sitter,
    determines the type annotation locations, splits the file at those
    locations, and uses the provided model to perform fill-in-the-middle to
    generate the missing type annotations.
    """
    def __init__(self, model):
        self.model = model
        self.language = Language(LANGUAGES_SO, "typescript")
        self.parser = Parser()
        self.parser.set_language(self.language)

    def parse(self, s: str):
        return self.parser.parse(s.encode("utf-8"))

    def node_to_str(self, node) -> str:
        return node.text.decode("utf-8")

    def run_query(self, content: str, query_str: str):
        tree = self.parse(content)
        query = self.language.query(query_str)
        return query.captures(tree.root_node)

    def convert_arrow_funs(self, content: str) -> str:
        """
        Ensures arrow functions have their parameters wrapped with parentheses.
        This allows type annotation prediction to be uniform (since type
        annotations for single-parameter arrow functions require the
        parentheses).

        Example:  x => x      becomes      (x) => x
        """
        captures = self.run_query(content,
            """
            (arrow_function parameter: (identifier) @param body: (_))
            """)
        pairs = [[c[0].start_byte, c[0].end_byte] for c in captures]

        # Given pairs of start and end bytes for the unparenthesized parameters,
        # iterate over the byte array in reverse order (so we don't offset the
        # indices) and insert ')' and '(' at the appropriate locations.
        content_bytes = bytearray(content.encode("utf-8"))
        for s, e in reversed(pairs):
            content_bytes[e:e] = bytearray(")".encode("utf-8"))
            content_bytes[s:s] = bytearray("(".encode("utf-8"))

        return content_bytes.decode("utf-8")

    def split_at_annotation_locations(self, content: str) -> list[str]:
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
        captures = self.run_query(content,
            """
            [
                (required_parameter pattern: (_) @ann)
                (optional_parameter pattern: (_)) @ann
                (formal_parameters) @ann
                (public_field_definition name: (_) @ann)
                (variable_declarator name: (_) @ann)
            ]
            """)

        # The end byte is where we want to insert type annotations.
        # We add None to the beginning and end of the list of indices, so we
        # can do array slicing, e.g. a[None:i] === a[:i]
        indices = [None] + sorted([c[0].end_byte for c in captures]) + [None]
        content_bytes = content.encode("utf-8")
        chunks = []

        # Zip the list with itself, offset by 1.
        # [None, i1, i2, None]    becomes    [(None, i1), (i1, i2), (i2, None)]
        # This gives us the chunks:           c[:i1],     c[i1:i2], c[i2:]
        # And we want to insert type annotations between the chunks.
        for s, e in zip(indices, indices[1:]):
            chunks.append(content_bytes[s:e].decode("utf-8"))

        return chunks

    def extract_type(self, generated: str) -> Optional[str]:
        """
        Given a string that represents the generated text from infilling,
        use the parser to extract the prefix that is a valid type annotation.

        This works by creating a variable declaration template, and the parser
        is lenient enough to extract the type annotation for a variable
        declarator, even if the entire string is not syntactically valid.

        Returns None if there is no valid type annotation.
        """
        template = f"let x: {generated}"
        captures = self.run_query(template,
            """
            (variable_declarator type: (type_annotation (_) @ann))
            """)
        return self.node_to_str(captures[0][0]) if captures else None

    def generate_type(self, prefix: str, suffix: str, retries: int = 3) -> str:
        """
        Generates a valid type annotation for the given prefix and suffix.
        The generated text may contain other code, so we only extract the prefix
        that is a valid type annotation. Tries up to retries times; otherwise
        returns "any".
        """
        for _ in range(retries):
            generated = self.model.infill(prefix, suffix)
            extracted = self.extract_type(generated)
            if extracted:
                return extracted
        return "any"

    def infill_types(self, chunks: list[str]) -> str:
        """
        Given a list of chunks, infills type annotations between those chunks.

        Assumes that a TypeScript file has been parsed and split up, so that the
        chunks contain the text in between the desired type annotations.

        Fills in one type at a time, from start to end. For a given type
        annotation location, the prefix contains all the previous chunks with
        the type annotations inserted, while the suffix concatenates the
        remaining chunks without type annotations.
        """
        if len(chunks) < 2:
            return "".join(chunks)

        infilled_prefix = chunks[0]
        for index, chunk in enumerate(chunks[1:]):
            infilled_prefix += ": "
            suffix = "".join(chunks[index + 1:])

            clipped_prefix, clipped_suffix = self.model.clip_text(infilled_prefix, suffix)
            filled_type = self.generate_type(clipped_prefix, clipped_suffix)
            infilled_prefix += filled_type + chunk

        return infilled_prefix

    def infer(self, content: str) -> str:
        """
        Run type inference, i.e. type annotation prediction, using
        fill-in-the-middle to infill types. Does not generate type definitions.
        """
        content = self.convert_arrow_funs(content)
        chunks = self.split_at_annotation_locations(content)
        return self.infill_types(chunks)

    def slice_input(self, content: str, slice_length: int) -> list[str]:
        """
        Splits the input into slices of slice_length.
        """
        slices = []
        for i in range(0, len(content), slice_length):
            slices.append(content[i : i + slice_length])
        return slices

    def infer_with_definitions(self, content: str) -> str:
        """
        Generate type annotations and definitions, by providing an instruction
        to the model.
        """
        # This instruction seems to work well. Need to mention "interfaces" to
        # get interfaces. Mentioning "type definitions" isn't enough, and
        # "classes" causes the model to generate additional classes and methods.
        # Mentioning "TypeScript" doesn't seem to work.
        instruction = "Add type annotations and interfaces"

        # max_tokens * estimated characters per token * 95% to leave some margin
        slice_length = int(self.model.max_tokens * 3.5 * 0.95)

        # TODO: Is there a better way of slicing the input to fit into the
        # context window? These slices can get the model to produce garbage.
        # And sometimes the output doesn't match the slice.
        result = []
        slices = self.slice_input(content, slice_length)
        for s in slices:
            result.append(self.model.edit(s, instruction))

        return "".join(result)
