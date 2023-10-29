from datasets import Dataset, IterableDataset
from pathlib import Path
from typing import Self
import argparse
import cmd
import datasets
import difflib
import numpy as np

import util

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RESET = "\033[39m"


def print_diff(
    before: str,
    after: str,
    fromfile: str = "before",
    tofile: str = "after",
    color: bool = False,
) -> None:
    before_lines = before.splitlines(keepends=True)
    after_lines = after.splitlines(keepends=True)
    diff = difflib.unified_diff(
        before_lines, after_lines, fromfile=fromfile, tofile=tofile
    )

    for d in diff:
        if color:
            if d.startswith("@@"):
                print(YELLOW + d, end=RESET)
            elif d.startswith("+"):
                print(GREEN + d, end=RESET)
            elif d.startswith("-"):
                print(RED + d, end=RESET)
        else:
            print(d, end="")
    print()


class Viewer(cmd.Cmd):
    intro = (
        "This is a shell for viewing StenoType experiment results.\n"
        "Type help or ? to list commands.\n"
    )
    prompt = "> "

    def __init__(self: Self, dataset: Dataset | IterableDataset):
        super().__init__()
        self.dataset = dataset

        self.problem_idx = 0
        self.completion_idx = 0
        self.total_problems = len(self.dataset)
        self.problem_completions = len(self.dataset[self.problem_idx]["results"])

        if not all(len(c.keys()) > 2 for r in self.dataset["results"] for c in r):
            print("error: dataset does not have results!")
            exit(2)

        # Pre-compute dataset summary
        self.total_completions = len([r for d in dataset for r in d["results"]])
        self.total_type_checks = np.sum(dataset["num_type_checks"])
        self.pct_type_checks = (
            0
            if self.total_completions == 0
            else self.total_type_checks / self.total_completions
        )
        self.avg_accuracy = np.mean(
            [r["accuracy"] for d in dataset for r in d["results"] if not r["error"]]
        )
        self.avg_levenshtein = np.mean(
            [r["levenshtein"] for d in dataset for r in d["results"] if not r["error"]]
        )
        self.avg_untyped_levenshtein = np.mean(
            [
                r["untyped_levenshtein"]
                for d in dataset
                for r in d["results"]
                if not r["error"]
            ]
        )
        self.avg_type_errors = np.mean(
            [r["type_errors"] for d in dataset for r in d["results"] if not r["error"]]
        )
        self.avg_parse_errors = np.mean(
            [r["parse_errors"] for d in dataset for r in d["results"] if not r["error"]]
        )
        self.pass_1 = np.mean(dataset["pass@1"])

        type_checks_problems = [
            str(i) for i, d in enumerate(dataset) if d["num_type_checks"] > 0
        ]
        self.type_checks_completions = {
            k: [
                f".{i}"
                for i, r in enumerate(dataset[int(k)]["results"])
                if r["type_checks"]
            ]
            for k in type_checks_problems
        }

        # Set up aliases
        self.aliases = {
            "n": self.do_next,
            "j": self.do_next,
            "prev": self.do_previous,
            "p": self.do_previous,
            "k": self.do_previous,
            "forw": self.do_forward,
            "f": self.do_forward,
            "l": self.do_forward,
            "back": self.do_backward,
            "b": self.do_backward,
            "h": self.do_backward,
            "g": self.do_goto,
            "s": self.do_summary,
            "c": self.do_code,
            "d": self.do_diff,
            "v": self.do_view,
            "q": self.do_quit,
        }

        # Print intro and summaries
        print(self.intro)
        self.do_summary("")

        # This loop is to handle Ctl-C (KeyboardInterrupt)
        while True:
            try:
                self.update_prompt()
                self.cmdloop(intro="")
                exit(0)
            except KeyboardInterrupt:
                print()

    def default(self, line: str):
        cmd, arg, line = self.parseline(line)
        if arg is None:
            arg = ""
        if cmd == "EOF":
            return self.do_quit(arg)
        if cmd in self.aliases:
            return self.aliases[cmd](arg)
        else:
            super().default(line)

    def update_prompt(self):
        # each problem may have different number of completions
        self.problem_completions = len(self.dataset[self.problem_idx]["results"])
        p_idx, c_idx = self.problem_idx, self.completion_idx
        p_tot, c_tot = self.total_problems, self.problem_completions

        example = self.dataset[self.problem_idx]
        name = example["max_stars_repo_name"] + " " + example["max_stars_repo_path"]

        self.prompt = (
            f"[problem {p_idx}/{p_tot - 1}]"
            f"[completion {c_idx}/{c_tot - 1}]"
            f"[{name}] \n"
            "> "
        )

    def problem_summary(self):
        """
        Show summary for the current problem.
        """
        example = self.dataset[self.problem_idx]

        print("===PROBLEM INFO===")
        print(
            f"Number of completions: {example['num_completions']}\n"
            f"Number type checks: {example['num_type_checks']} "
            f"({example['pct_type_checks']:.1%})\n"
            f"Average accuracy: {example['avg_accuracy']:.1%}\n"
            f"Average Levenshtein: {example['avg_levenshtein']:.1%}\n"
            f"Average untyped Levenshtein: {example['avg_untyped_levenshtein']:.1%}\n"
            f"Average type errors: {example['avg_type_errors']:.1f}\n"
            f"Average parse errors: {example['avg_parse_errors']:.1f}\n"
            f"pass@1: {example['pass@1']:.1%}"
        )
        if str(self.problem_idx) in self.type_checks_completions:
            print()
            completions = self.type_checks_completions[str(self.problem_idx)]
            print("Completions that type check:")
            print(GREEN + " ".join(completions) + RESET)

    def completion_summary(self):
        """
        Show information for the current problem/completion stats.
        """
        example = self.dataset[self.problem_idx]
        completion = example["results"][self.completion_idx]

        print("===COMPLETION INFO===")
        print(
            f"Accuracy: {completion['accuracy']:.1%}\n"
            f"Levenshtein: {completion['levenshtein']:.1%}\n"
            f"Untyped Levenshtein: {completion['untyped_levenshtein']:.1%}\n"
            f"Type errors: {completion['type_errors']}\n"
            f"Parse errors: {completion['parse_errors']}\n"
            f"Type checks: ",
            end="",
        )
        if completion["type_checks"]:
            print(GREEN + "YES" + RESET)
        else:
            print(RED + "NO" + RESET)

    def postcmd(self, stop: bool, line: str) -> bool:
        self.update_prompt()
        return stop

    def do_next(self, arg: str):
        """
        Move to the next problem.

        aliases: n, j
        """
        if self.problem_idx == self.total_problems - 1:
            print("error: reached end of problem list")
        else:
            self.problem_idx += 1
            self.completion_idx = 0
            self.problem_summary()
            self.completion_summary()

    def do_previous(self, arg: str):
        """
        Move to the previous problem.

        aliases: prev, p, k
        """
        if self.problem_idx == 0:
            print("error: reached beginning of problem list")
        else:
            self.problem_idx -= 1
            self.completion_idx = 0
            self.problem_summary()
            self.completion_summary()

    def do_forward(self, arg: str):
        """
        Move forward in the completions list (for the current problem).

        aliases: forw, f, l
        """
        if self.completion_idx == self.problem_completions - 1:
            if self.problem_idx == self.total_problems - 1:
                print(
                    "error: reached end of completions list and cannot wrap "
                    "around to next problem"
                )
            else:
                self.problem_idx += 1
                self.completion_idx = 0
                self.problem_summary()
                self.completion_summary()
                print("wrapping around to next problem")
        else:
            self.completion_idx += 1
            self.completion_summary()

    def do_backward(self, arg: str):
        """
        Move backward in the completions list (for the current problem).

        aliases: back, b, h
        """
        if self.completion_idx == 0:
            if self.problem_idx == 0:
                print(
                    "error: reached beginning of completions list and cannot "
                    "wrap around to previous problem"
                )
            else:
                self.problem_idx -= 1
                self.completion_idx = len(self.dataset[self.problem_idx]["results"]) - 1
                self.problem_summary()
                self.completion_summary()
                print("wrapping around to previous problem")
        else:
            self.completion_idx -= 1
            self.completion_summary()

    def do_goto(self, arg: str):
        """
        Jump to the given problem and completion, given in the format
        "[problem index].[completion index]".

        Examples:
          goto 1.3      jumps to problem index 1 and completion index 3
          goto 1.       equivalent to goto 1.0
          goto 1        equivalent to goto 1.0
          goto .3       equivalent to goto [current problem index].3

        aliases: g
        """
        if not arg:
            print("error: no indices given")
            self.do_help("goto")
            return

        error = False
        old_p_idx = self.problem_idx
        old_c_idx = self.completion_idx
        indices = arg.split(".")
        try:
            if len(indices) == 1:
                new_idx = int(indices[0])
                if 0 <= new_idx and new_idx < self.total_problems:
                    self.problem_idx = new_idx
                    self.completion_idx = 0
                else:
                    error = True
            elif len(indices) == 2:
                if indices[0]:
                    p_idx = int(indices[0])
                    if 0 <= p_idx and p_idx < self.total_problems:
                        self.problem_idx = p_idx
                        self.problem_completions = len(self.dataset[p_idx]["results"])
                    else:
                        error = True
                if indices[1]:
                    c_idx = int(indices[1])
                    if 0 <= c_idx and c_idx < self.problem_completions:
                        self.completion_idx = c_idx
                    else:
                        error = True
            else:
                error = True
        except ValueError:
            error = True

        if error:
            self.problem_idx = old_p_idx
            self.completion_idx = old_c_idx
            print("error: invalid indices given")
            self.do_help("goto")
        self.problem_summary()
        self.completion_summary()

    def do_summary(self, arg: str):
        """
        Show summary of overall dataset, as well as the current problem/completion.

        aliases: s
        """
        print("===DATASET SUMMARY===")
        print(
            f"Total completions: {self.total_completions}\n"
            f"Total type checks: {self.total_type_checks} "
            f"({self.pct_type_checks:.1%})\n"
            f"Average accuracy: {self.avg_accuracy:.1%}\n"
            f"Average Levenshtein: {self.avg_levenshtein:.1%}\n"
            f"Average untyped Levenshtein: {self.avg_untyped_levenshtein:.1%}\n"
            f"Average type errors: {self.avg_type_errors:.1f}\n"
            f"Average parse errors: {self.avg_parse_errors:.1f}\n"
            f"pass@1: {self.pass_1:.1%}"
        )
        if self.type_checks_completions:
            print()
            completions = [
                k
                for k, _ in sorted(
                    self.type_checks_completions.items(), key=lambda v: len(v[1])
                )
            ]
            print(
                "Problems that type check (in ascending order of completions "
                "that type check):"
            )
            print(GREEN + " ".join(completions) + RESET)
        print()

        self.problem_summary()
        self.completion_summary()

    def do_code(self, arg: str):
        """
        Print the code for the current problem/completion.
        Prints the original code, the input code (i.e. code with no types), and
        the output.

        aliases: c
        """
        example = self.dataset[self.problem_idx]
        completion = example["results"][self.completion_idx]
        original_code = example["content"]
        input_code = example["content_without_types"]
        output_code = completion["output"]

        print("===ORIGINAL===")
        print(original_code)
        print("===INPUT===")
        print(input_code)
        print("===OUTPUT===")
        print(output_code)

    def do_diff(self, arg: str):
        """
        Print the diffs for original/output and input/output.

        aliases: d
        """
        example = self.dataset[self.problem_idx]
        completion = example["results"][self.completion_idx]
        original_code = example["content"]
        input_code = example["content_without_types"]
        output_code = completion["output"]

        print("===DIFF ORIGINAL/OUTPUT===")
        print_diff(
            original_code, output_code, fromfile="original", tofile="output", color=True
        )
        print("===DIFF INPUT/OUTPUT===")
        print_diff(
            input_code, output_code, fromfile="input", tofile="output", color=True
        )

    def do_view(self, arg: str):
        """
        Show the current problem/completion and diffs.

        aliases: v
        """
        self.do_code(arg)
        self.do_diff(arg)
        self.completion_summary()

    def do_quit(self, arg: str) -> bool:
        """
        Quit the viewer.
        aliases: q
        """
        print("Exiting...")
        return True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Results viewer for StenoType")

    parser.add_argument(
        "--dataset", type=str, required=True, help="path to dataset to view"
    )

    args = parser.parse_args()

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print("error: cannot find dataset:", dataset_path)
        exit(2)

    return args


def main():
    # Don't cache datasets
    datasets.disable_caching()

    args = parse_args()
    dataset = util.load_dataset(args.dataset)

    Viewer(dataset).cmdloop()


if __name__ == "__main__":
    main()
