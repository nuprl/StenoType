from pathlib import Path
from text_generation import Client

ENDPOINT_FILE = ".STARCODER_ENDPOINT"

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
    def __init__(
        self,
        max_fim_tokens: int = 50,
        max_tokens: int = 2048,
        temperature: float = 0.2,
        top_p: float = 0.95,
        max_context_length: int = 500,
        timeout: int = 60
    ):
        self.max_fim_tokens = max_fim_tokens
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_context_length = max_context_length

        if not Path(ENDPOINT_FILE).exists():
            print("Unknown API endpoint; make sure .STARCODER_ENDPOINT exists and contains the endpoint URL.")
            exit(2)
        endpoint = Path(ENDPOINT_FILE).read_text().strip()

        self.client = Client(endpoint, timeout=timeout)

    def prefix_ending_with_newline(self, s: str, max_length: int) -> str:
        """
        Produces a prefix of s that is at most max_length, but does not split a
        line.
        """
        return s[:max_length].rsplit("\n", 1)[0]

    def suffix_starting_with_newline(self, s: str, max_length: int) -> str:
        """
        Produces a suffix of s that is at most max_length, but does not split a
        line.
        """
        return s[-max_length:].split("\n", 1)[-1]

    def clip_text(self, s1: str, s2: str) -> tuple[str, str]:
        """
        Clips s1 and s2 so that each string is at most half of
        max_context_length (which is measured in characters).
        """
        if len(s1) < self.max_context_length // 2:
            # s1 is short enough, so get the longest prefix of s2
            s2 = self.prefix_ending_with_newline(s2, self.max_context_length - len(s1))
        elif len(s2) < self.max_context_length // 2:
            # s2 is short enough, so get the longest suffix of s1
            s1 = self.suffix_starting_with_newline(s1, self.max_context_length - len(s2))
        else:
            # Both strings are too long
            s1 = self.suffix_starting_with_newline(s1, self.max_context_length // 2)
            s2 = self.prefix_ending_with_newline(s2, self.max_context_length // 2)
        return s1, s2

    def generate(self, prompt: str, **kwargs) -> str:
        """
        Call the model to generate a completion. Use a default configuration
        that can be overridden with keyword arguments.
        """
        config = {
            "do_sample": True,
            "max_new_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }
        config.update(kwargs)
        output = self.client.generate(prompt, **config).generated_text
        return output.removesuffix(ENDOFTEXT)

    def infill(self, prefix: str, suffix: str) -> str:
        prompt = f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"
        return self.generate(prompt, max_new_tokens=self.max_fim_tokens)

    def edit(self, code: str, instruction: str, prefix: str = "") -> str:
        prompt = f"{COMMIT_BEFORE}{code}{COMMIT_MSG}{instruction}{COMMIT_AFTER}{prefix}"
        return self.generate(prompt)
