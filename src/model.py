from pathlib import Path
from requests.exceptions import ReadTimeout
from text_generation import Client
from transformers import AutoTokenizer
from typing import Optional, Self
import shutil
import subprocess
import time

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
    infilling. Must be used as a context manager.
    """
    CONTEXT_SIZE = 8 * 1024 # 8K tokens
    TYPE_PROPORTION = 0.25  # Assume input expands by 25% (added types) to get output

    # Input + Output = Context; Output = (1 + Proportion) * Input
    # Input + (1 + Proportion) * Input = Context
    # (2 + Proportion) * Input = Context
    # Input = Context / (2 + Proportion)
    INPUT_SIZE = int(CONTEXT_SIZE / (2 + TYPE_PROPORTION))
    OUTPUT_SIZE = CONTEXT_SIZE - INPUT_SIZE

    finalized = False

    def __init__(
        self,
        model_path: str,
        port: int,
        devices: str,
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

        self.tokenizer = AutoTokenizer.from_pretrained(Path(model_path).resolve())
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

        self.model_path = model_path
        self.port = port
        self.devices = devices
        self.timeout = timeout

    def __enter__(self) -> Self:
        print(f"Starting text-generation-inference server for {self.model_path} ...")

        # Try podman, then docker
        container_exec = shutil.which("podman") or shutil.which("docker")
        if container_exec is None:
            raise RuntimeError("Either podman or docker must be installed")

        model_paths = Path(self.model_path).resolve().parts
        models_directory = str(Path(*model_paths[:-1]))
        model_name = model_paths[-1]

        args = [
            container_exec, "run",
            "-p", f"{self.port}:80",
            "-v", f"{models_directory}:/data",
            "-e", f"NVIDIA_VISIBLE_DEVICES={self.devices}",
            "-e", "HF_HUB_ENABLE_HF_TRANSFER=0",
            "ghcr.io/huggingface/text-generation-inference:0.8",
            "--model-id", f"/data/{model_name}",
            "--max-input-length", "8192",
            "--max-total-tokens", "16384",
            "--max-waiting-tokens", "65536"
        ]
        self.server_process: subprocess.Popen = subprocess.Popen(
            args, stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, encoding="utf-8"
        )

        # Hack: just sleep for 10 seconds and hope the server has started
        time.sleep(10)
        self.client = Client(f"http://127.0.0.1:{self.port}", timeout=self.timeout)

        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        print("Shutting down text-generation-inference server...")
        self.server_process.terminate()
        self.finalized = True
        # Hack: sleep for 10 seconds before returning
        time.sleep(10)

    def _generate(self, prompt: str, **kwargs) -> Optional[str]:
        """
        Call the model to generate a completion. Use a default configuration
        that can be overridden with keyword arguments.
        """
        if self.finalized:
            raise RuntimeError("Cannot generate completion after Model has "
                               "been finalized")

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
