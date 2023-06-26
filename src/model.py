from pathlib import Path
from requests.exceptions import ReadTimeout
from text_generation import Client
from transformers import AutoTokenizer
from typing import Optional

from util import ROOT_DIR

ENDPOINT_FILE = str(Path(ROOT_DIR, ".STARCODER_ENDPOINT").resolve())
MODEL_PATH = str(Path(ROOT_DIR.parent, "models", "starcoderbase").resolve())

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
COMMIT_BEFORE = "<commit_before>"
COMMIT_MSG = "<commit_msg>"
COMMIT_AFTER = "<commit_after>"
ENDOFTEXT = "<|endoftext|>"

class Model:
    """
    This class wraps the text_generation client and provides methods for
    infilling.
    """
    CONTEXT_SIZE = 8 * 1024 # 8K tokens
    TYPE_PROPORTION = 0.25  # Assume input expands by 25% (added types) to get output

    # Input + Output = Context; Output = (1 + Proportion) * Input
    # Input + (1 + Proportion) * Input = Context
    # (2 + Proportion) * Input = Context
    # Input = Context / (2 + Proportion)
    INPUT_SIZE = int(CONTEXT_SIZE / (2 + TYPE_PROPORTION))
    OUTPUT_SIZE = CONTEXT_SIZE - INPUT_SIZE

    def __init__(
        self,
        max_fim_tokens: int = 50,
        max_new_tokens: int = OUTPUT_SIZE,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_context_length: int = 500,
        timeout: int = 600,
    ):
        self.max_fim_tokens = max_fim_tokens
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_length = max_context_length

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        if not Path(ENDPOINT_FILE).exists():
            print("Unknown API endpoint; make sure .STARCODER_ENDPOINT exists "
                  "and contains the endpoint URL.")
            exit(2)
        endpoint = Path(ENDPOINT_FILE).read_text().strip()

        self.client = Client(endpoint, timeout=timeout)

    def _generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Call the model to generate a completion. Use a default configuration
        that can be overridden with keyword arguments.
        """
        config = {
            "do_sample": True,
            "max_new_tokens": self.max_new_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        config.update(kwargs)
        try:
            output = self.client.generate(prompt, **config).generated_text
            return output.removesuffix(ENDOFTEXT)
        except ReadTimeout:
            return None

    def tokenize(self, content: str):
        # Assuming NumPy tensors consume less memory
        return self.tokenizer(content,
                              return_attention_mask=False,
                              return_tensors="np")["input_ids"][0]

    def infill(self, prefix: str, suffix: str) -> Optional[str]:
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        return self._generate(prompt, max_new_tokens=self.max_fim_tokens)

    def edit(self, code: str, instruction: str, prefix: str = "") -> Optional[str]:
        prompt = f"{COMMIT_BEFORE}{code}{COMMIT_MSG}{instruction}{COMMIT_AFTER}{prefix}"
        return self._generate(prompt)
