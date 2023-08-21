from tqdm import tqdm
from typing import Optional

from model import Model
from util import transform

# This instruction seems to work well. Need to mention "interfaces" to
# get interfaces. Mentioning "type definitions" isn't enough, and
# "classes" causes the model to generate additional classes and methods.
# Mentioning "TypeScript" doesn't seem to work.
DEFAULT_INSTRUCTION = "Add type annotations and interfaces"

class TypeInference:
    """
    Performs type inference by predictiong type annotations, and optionally
    generating type definitions.
    """
    def __init__(self, model: Model, instruction: str = DEFAULT_INSTRUCTION):
        self.model = model
        self.instruction = instruction

    def _extract_type(self, generated: str) -> Optional[str]:
        """
        Given a string that represents the generated text from infilling,
        use the parser to extract the prefix that is a valid type annotation.

        This works by creating a variable declaration template, and the parser
        is lenient enough to extract the type annotation for a variable
        declarator, even if the entire string is not syntactically valid.

        Returns None if there is no valid type annotation.
        """
        template = f"let x: {generated}"
        captures = transform.run_query(template,
            """
            (variable_declarator type: (type_annotation (_) @ann))
            """)
        return transform.node_to_str(captures[0][0]) if captures else None

    def _generate_type(self, prefix: str, suffix: str, retries: int = 3) -> str:
        """
        Generates a valid type annotation for the given prefix and suffix.
        The generated text may contain other code, so we only extract the prefix
        that is a valid type annotation. Tries up to retries times; otherwise
        returns "any".
        """
        for _ in range(retries):
            generated = self.model.infill(prefix, suffix)
            extracted = self._extract_type(generated) if generated else None
            if extracted:
                return extracted
        return "any"

    def _infill_types(self, chunks: list[str]) -> str:
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
        for index, chunk in tqdm(enumerate(chunks[1:]),
                                 desc="Infilling",
                                 total=len(chunks)-1,
                                 leave=False):
            infilled_prefix += ": "
            suffix = "".join(chunks[index + 1:])

            clipped_prefix, clipped_suffix = transform.clip_text(
                infilled_prefix, suffix, self.model.max_context_length
            )
            filled_type = self._generate_type(clipped_prefix, clipped_suffix)
            infilled_prefix += filled_type + chunk

        return infilled_prefix

    def infer(self, content: str) -> str:
        """
        Run type inference, i.e. type annotation prediction, using
        fill-in-the-middle to infill types. Does not generate type definitions.
        """
        content = transform.parenthesize_arrow_params(content)
        chunks = transform.split_at_annotation_locations(content)
        return self._infill_types(chunks)

    def infer_with_definitions(self, content: str) -> Optional[str]:
        """
        Generate type annotations and definitions, by providing an instruction
        to the model.
        """
        # Assume no slicing is needed
        return self.model.edit(content, self.instruction)
