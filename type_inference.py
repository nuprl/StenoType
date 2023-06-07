from pathlib import Path
from tree_sitter import Language, Parser, Node, Tree

LANGUAGES_SO = f"{Path(__file__).parent}/build/languages.so"
TREE_SITTER_TS = f"{Path(__file__).parent}/tree-sitter-typescript/typescript"

Language.build_library(LANGUAGES_SO, [TREE_SITTER_TS])

class TypeInference:
    def __init__(self, model):
        self.model = model
        self.parser = Parser()
        self.language = Language(LANGUAGES_SO, "typescript")
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

    def infer(self, content: str):
        content = self.convert_arrow_funs(content)
        chunks = self.split_at_annotation_locations(content)

        # TODO
        # Now we have chunks split at type annotation locations
        # So we can construct prefix/suffix and use that for infilling
        # Still need to figure out clipping (if it's too long)
        # And maybe parsing the returned type

        print(content)
