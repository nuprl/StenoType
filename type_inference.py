from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree
from typing import Optional

LANGUAGES_SO = f"{Path(__file__).parent}/build/languages.so"
TREE_SITTER_TS = f"{Path(__file__).parent}/tree-sitter-typescript/typescript"

Language.build_library(LANGUAGES_SO, [TREE_SITTER_TS])

class TypeInference:
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
        Ensures arrow functions have their parameters wrapped with parentheses, e.g.:
        Before: x => x
        After: (x) => x
        """
        captures = self.run_query(content,
            """
            (arrow_function parameter: (identifier) @param body: (_))
            """)
        pairs = [[c[0].start_byte, c[0].end_byte] for c in captures]

        content_bytes = bytearray(content.encode("utf-8"))
        for s, e in reversed(pairs):
            content_bytes[e:e] = bytearray(")".encode("utf-8"))
            content_bytes[s:s] = bytearray("(".encode("utf-8"))

        return content_bytes.decode("utf-8")

    def split_at_annotation_locations(self, content: str) -> list[str]:
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
        indices = [None] + sorted([c[0].end_byte for c in captures]) + [None]
        content_bytes = content.encode("utf-8")

        chunks = []
        for s, e in zip(indices, indices[1:]):
            chunks.append(content_bytes[s:e].decode("utf-8"))

        return chunks

    def extract_type(self, generated_type: str) -> Optional[str]:
        template = f"let x: {generated_type}"
        captures = self.run_query(template,
            """
            (variable_declarator type: (type_annotation (_) @ann))
            """)
        return self.node_to_str(captures[0][0]) if captures else None

    def generate_type(self, prefix: str, suffix: str, retries: int = 3) -> str:
        for _ in range(retries):
            generated = self.model.infill(prefix, suffix)
            extracted = self.extract_type(generated)
            if extracted:
                return extracted
        return "any"

    def infill_types(self, chunks: list[str]) -> str:
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
        content = self.convert_arrow_funs(content)
        chunks = self.split_at_annotation_locations(content)
        return self.infill_types(chunks)
