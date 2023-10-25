from pathlib import Path
from transformers import AutoTokenizer
from typing import Optional
from vllm import LLM, SamplingParams
import torch

FIM_PREFIX = "<fim_prefix>"
FIM_MIDDLE = "<fim_middle>"
FIM_SUFFIX = "<fim_suffix>"
COMMIT_BEFORE = "<commit_before>"
COMMIT_MSG = "<commit_msg>"
COMMIT_AFTER = "<commit_after>"
ENDOFTEXT = "<|endoftext|>"


class Tokenizer:
    """
    Wrapper for an AutoTokenizer, because we need to add a special "[PAD]" token.
    For convenience, this class can be called to tokenize a string.
    """

    def __init__(self, tokenizer_path: str):
        self.tokenizer = AutoTokenizer.from_pretrained(Path(tokenizer_path).resolve())
        self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    def __call__(self, content: str):
        # Assuming NumPy tensors consume less memory
        return self.tokenizer(
            content, return_attention_mask=False, return_tensors="np"
        )["input_ids"][0]


class Model:
    """
    This class wraps vLLM and provides methods for completion, infilling, and
    editing (via the StarCoder git commit format).
    """

    CONTEXT_SIZE = 8 * 1024  # 8K tokens
    TYPE_PROPORTION = 0.25  # Assume input expands by 25% (added types) to get output

    # Input + Output = Context; Output = (1 + Proportion) * Input
    # Input + (1 + Proportion) * Input = Context
    # (2 + Proportion) * Input = Context
    # Input = Context / (2 + Proportion)
    INPUT_SIZE = round(CONTEXT_SIZE / (2 + TYPE_PROPORTION))
    OUTPUT_SIZE = CONTEXT_SIZE - INPUT_SIZE

    def __init__(
        self,
        model_path: str,
        max_tokens: int = OUTPUT_SIZE,
        max_fim_tokens: int = 50,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_context_length: int = 500,
    ):
        self.max_tokens = max_tokens
        self.max_fim_tokens = max_fim_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_length = max_context_length

        self.model = LLM(
            model=model_path,
            tokenizer=model_path,
            dtype="bfloat16" if torch.cuda.is_bf16_supported() else "float16",
        )

    def _generate(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Call the model to generate a completion. Use a default configuration
        that can be overridden with keyword arguments. See vLLM documentation
        for all configuration options:
        https://github.com/vllm-project/vllm/blob/v0.2.0/vllm/sampling_params.py#L15

        `n` is not allowed as a config option, because this method returns a
        single completion for each prompt. To generate multiple completions for
        a prompt, call this method with multiple copies of the prompt.
        """
        prompts = [prompt.strip() for prompt in prompts]
        config = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }
        config.update(kwargs)
        if "n" in config:
            del config["n"]
        params = SamplingParams(**config)

        outputs = self.model.generate(prompts, params, use_tqdm=False)
        return [o.outputs[0].text for o in outputs]

    def infill_batch(self, pairs: list[tuple[str, str]]) -> list[str]:
        prompts = [
            f"{FIM_PREFIX}{pre}{FIM_SUFFIX}{suf}{FIM_MIDDLE}" for pre, suf in pairs
        ]
        return self._generate(prompts, max_tokens=self.max_fim_tokens)

    def infill(self, prefix: str, suffix: str) -> str:
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        return self._generate([prompt], max_tokens=self.max_fim_tokens)[0]

    def edit_batch(self, triples: list[tuple[str, str]] | list[tuple[str, str, str]]):
        # prefix may be missing, so we unpack as a list and treat it as an empty string
        prompts = [
            f"{COMMIT_BEFORE}{code}"
            f"{COMMIT_MSG}{instruction}"
            f"{COMMIT_AFTER}{''.join(prefix)}"
            for code, instruction, *prefix in triples
        ]
        return self._generate(prompts)

    def edit(self, code: str, instruction: str, prefix: str = "") -> str:
        prompt = f"{COMMIT_BEFORE}{code}{COMMIT_MSG}{instruction}{COMMIT_AFTER}{prefix}"
        return self._generate([prompt])[0]

    def complete_batch(
        self, prompts: list[str], stop: Optional[list[str]] = None
    ) -> list[str]:
        completions = self._generate(prompts, stop=stop)
        return [
            f"{prompt}{completion}" for prompt, completion in zip(prompts, completions)
        ]

    def complete(self, prompt: str, stop: Optional[list[str]] = None) -> str:
        completion = self._generate([prompt], stop=stop)[0]
        return f"{prompt}{completion}"
