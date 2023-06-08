from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree

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

    def run_query(self, tree, query_str: str):
        query = self.language.query(query_str)
        return query.captures(tree.root_node)

    def convert_arrow_funs(self, content: str) -> str:
        """
        Ensures arrow functions have their parameters wrapped with parentheses, e.g.:
        Before: x => x
        After: (x) => x
        """
        tree = self.parse(content)
        captures = self.run_query(tree,
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
        tree = self.parse(content)
        captures = self.run_query(tree,
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

    def infill_types(self, chunks: list[str]) -> str:
        if len(chunks) < 2:
            return "".join(chunks)

        infilled_prefix = chunks[0]
        for index, chunk in enumerate(chunks[1:]):
            infilled_prefix += ": "
            suffix = "".join(chunks[index + 1:])

            clipped_prefix, clipped_suffix = self.model.clip_text(infilled_prefix, suffix)
            filled_type = self.model.infill(clipped_prefix, clipped_suffix)
            print("Prefix:", clipped_prefix)
            print("Type:", filled_type)
            print("Suffix:", clipped_suffix)
            print()

            infilled_prefix += filled_type + chunk

        return infilled_prefix

    def infer(self, content: str):
        content = self.convert_arrow_funs(content)
        chunks = self.split_at_annotation_locations(content)
        
        infilled = self.infill_types(chunks)

        # TODO
        # Need to parse and extract the returned type

        print(infilled)
