from pathlib import Path
from typing import Any, Callable, Self
import argparse
import cmd
import datasets
import difflib
import os
import re

import util

RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
BLUE = "\033[34m"
MAGENTA = "\033[35m"
CYAN = "\033[36m"
RESET = "\033[39m"

INDEX_SPEC_RE = re.compile(r"^(\d+\.\d+|\d+\.|\d+|\.\d+)$")


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
        else:
            print(d, end="")
    print()


class Viewer(cmd.Cmd):
    intro = (
        "This is a shell for viewing StenoType experiment results.\n"
        "Type help or ? to list commands.\n"
    )
    prompt = "> "

    def __init__(self: Self, dataset_path: str, summary_path: str):
        super().__init__()
        dataset = util.load_dataset(dataset_path)

        config_name = Path(dataset_path).stem
        summary = [
            j for j in util.read_jsonl(summary_path) if j["model"] == config_name
        ]
        if not summary:
            print(f"error: could not find config {config_name} in summary.jsonl!")
            exit(2)

        self.dataset = dataset
        self.summary = summary[0]

        self.problem_idx = 0
        self.completion_idx = 0
        self.total_problems = len(dataset)
        self.problem_completions = len(dataset[self.problem_idx]["results"])

        # Pre-compute dataset summary
        correct_problems = [
            i
            for i, d in enumerate(dataset)
            if len([r for r in d["results"] if r["correct"]]) > 0
        ]
        self.correct_completions = {
            str(k): [
                str(i) for i, r in enumerate(dataset[k]["results"]) if r["correct"]
            ]
            for k in correct_problems
        }

        # Set up aliases
        self.aliases: dict[str, Callable[[str], Any]] = {
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
        if INDEX_SPEC_RE.match(line):
            return self.do_goto(line)
        else:
            super().default(line)

    def update_prompt(self):
        # each problem may have different number of completions
        self.problem_completions = len(self.dataset[self.problem_idx]["results"])
        p_idx, c_idx = self.problem_idx, self.completion_idx
        p_tot, c_tot = self.total_problems, self.problem_completions

        name = self.dataset[self.problem_idx]["name"]

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
        summary = example["summaries"]

        print()
        print("===PROBLEM INFO===")
        print(
            f"{summary['num_completions']} completions "
            f"({summary['pct_correct']:.1%} correct)\n"
            f"{summary['pct_pkg_parses']:.1%} parse and "
            f"{summary['pct_type_checks']:.1%} type check\n"
            f"{summary['avg_errors']:.1f} errors per problem and "
            f"{summary['errors_per_file']:.1f} errors per file\n"
            f"{summary['avg_levenshtein']:.1%} Levenshtein and "
            f"{summary['avg_untyped_levenshtein']:.1%} untyped Levenshtein\n"
            f"{summary['avg_annotations_added']:.1f} annotations added "
            f"({summary['pct_annotation_sites_filled']:.1%} sites filled and "
            f"{summary['pct_annotations_trivial']:.1%} trivial annotations)\n"
            f"{summary['avg_definitions_added']:.1f} definitions added and "
            f"{summary['avg_definitions_used']:.1f} definitions used "
            f"({summary['avg_types_undefined']:.1f} types not defined)\n"
            f"{BLUE}pass@1 (project): {summary['pass@1_project']:.1%}{RESET}\n"
            f"{BLUE}pass@1 (files): {summary['pass@1_files']:.1%}{RESET}\n",
            end="",
        )
        if str(self.problem_idx) in self.correct_completions:
            print()
            completions = self.correct_completions[str(self.problem_idx)]
            dedup = {example["results"][int(i)]["output"]: [] for i in completions}
            for c in completions:
                output = example["results"][int(c)]["output"]
                dedup[output].append(f".{c}")

            print(
                f"Completions that are correct ({len(dedup)} unique, on separate lines):"
            )
            print(GREEN, end="")
            for dlist in dedup.values():
                print(" ".join(dlist))
            print(RESET, end="")

    def completion_summary(self):
        """
        Show information for the current problem/completion stats.
        """
        example = self.dataset[self.problem_idx]
        completion = example["results"][self.completion_idx]

        parses = f"{GREEN}YES{RESET}" if completion["pkg_parses"] else f"{RED}NO{RESET}"
        typechecks = (
            f"{GREEN}YES{RESET}" if completion["type_checks"] else f"{RED}NO{RESET}"
        )
        correct = f"{GREEN}YES{RESET}" if completion["correct"] else f"{RED}NO{RESET}"

        print()
        print("===COMPLETION INFO===")
        print(
            f"{completion['num_files']} files "
            f"({completion['num_correct_files']} correct)\n"
            f"{completion['num_errors']} errors and "
            f"{completion['errors_per_file']} errors per file\n"
            f"{completion['levenshtein']:.1%} Levenshtein and "
            f"{completion['untyped_levenshtein']:.1%} untyped Levenshtein\n"
            f"{completion['num_annotations_added']} annotations added "
            f"({completion['pct_annotation_sites_filled']:.1%} sites filled and "
            f"{completion['pct_annotations_trivial']:.1%} trivial annotations)\n"
            f"{completion['num_definitions_added']} definitions added "
            f"{completion['num_definitions_used']} definitions used "
            f"({completion['num_types_undefined']} types not defined)\n"
            f"Type annotations: {CYAN}{completion['type_annotations']}{RESET}\n"
            f"Type definitions: {CYAN}{completion['type_definitions']}{RESET}\n"
            f"Type definitions used: {CYAN}{completion['type_definitions_used']}{RESET}\n"
            f"Types not defined: {CYAN}{completion['types_undefined']}{RESET}\n"
            f"parses ({parses}) and type checks ({typechecks}) and is correct ({correct})\n"
        )
        return
        print(
            f"Accuracy: {completion['accuracy']:.1%}\n"
            f"Levenshtein: {completion['levenshtein']:.1%}\n",
            end="",
        )
        if completion["untyped_levenshtein"]:
            print(
                f"Untyped Levenshtein (for files that type check): {completion['untyped_levenshtein']:.1%}\n",
                end="",
            )
        print(
            f"Type errors: {completion['type_errors']}\n"
            f"Parse errors: {completion['parse_errors']}\n"
            f"Type checks: {completion['type_checks']}\n"
            "Correct: ",
            end="",
        )
        if completion["correct"]:
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
          3.4           equivalent to goto 3.4

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
        summary = self.summary
        print("===DATASET SUMMARY===")
        print(
            f"{summary['tot_completions']} completions across "
            f"{summary['num_problems']} problems\n"
            f"{summary['pct_pkg_parses']:.1%} parse and "
            f"{summary['pct_type_checks']:.1%} type check\n"
            f"{summary['avg_errors']:.1f} errors per problem and "
            f"{summary['errors_per_file']:.1f} errors per file\n"
            f"{summary['avg_levenshtein']:.1%} Levenshtein and "
            f"{summary['avg_untyped_levenshtein']:.1%} untyped Levenshtein\n"
            f"{summary['avg_annotations_added']:.1f} annotations added "
            f"({summary['pct_annotation_sites_filled']:.1%} sites filled and "
            f"{summary['pct_annotations_trivial']:.1%} trivial annotations)\n"
            f"{summary['avg_definitions_added']:.1f} definitions added and "
            f"{summary['avg_definitions_used']:.1f} definitions used "
            f"({summary['avg_types_undefined']:.1f} types not defined)\n"
            f"{BLUE}pass@1 (project): {summary['pass@1_project']:.1%}{RESET}\n"
            f"{BLUE}pass@1 (files): {summary['pass@1_files']:.1%}{RESET}\n",
            end="",
        )

        if self.correct_completions:
            print()
            print("Correct problems: ", end="")
            print(GREEN + " ".join(self.correct_completions.keys()) + RESET)

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
    results_path = Path(util.ROOT_DIR, "results").resolve()
    results_relpath = os.path.relpath(results_path)

    parser = argparse.ArgumentParser(description="Results viewer for StenoType")

    parser.add_argument(
        "--results_directory",
        type=str,
        default=results_path,
        help=f"directory containing results and summary; defaults to {results_relpath}",
    )
    parser.add_argument(
        "--dataset", type=str, required=True, help="path to dataset to view"
    )

    args = parser.parse_args()

    results_directory = Path(args.results_directory).resolve()
    args.results_directory = str(results_directory)
    if not results_directory.exists():
        print("error: cannot find results directory:", results_directory)
        exit(2)

    jsonl_path = Path(args.results_directory, "summary.jsonl")
    args.summary = jsonl_path
    if not jsonl_path.exists():
        print("error: cannot find summary file:", jsonl_path)
        exit(2)

    dataset_path = Path(args.dataset).resolve()
    if not dataset_path.exists():
        print("error: cannot find dataset:", dataset_path)
        exit(2)

    return args


def main():
    # Don't cache datasets
    datasets.disable_caching()
    args = parse_args()
    Viewer(args.dataset, args.summary).cmdloop()


if __name__ == "__main__":
    main()
