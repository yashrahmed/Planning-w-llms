# TODO: Define actions and use a solver.

import argparse
from dataclasses import dataclass

from unified_planning.shortcuts import BoolType, Fluent, Problem


@dataclass(frozen=True)
class State:
    priests_left: int
    cannibals_left: int
    boat_on_left: bool

    def priests_right(self, n: int) -> int:
        return n - self.priests_left

    def cannibals_right(self, n: int) -> int:
        return n - self.cannibals_left

    def is_safe(self, n: int) -> bool:
        left_bank_is_safe = (
            self.priests_left == 0 or self.priests_left >= self.cannibals_left
        )
        right_bank_is_safe = (
            self.priests_right(n) == 0
            or self.priests_right(n) >= self.cannibals_right(n)
        )
        return left_bank_is_safe and right_bank_is_safe

    @property
    def fluent(self) -> Fluent:
        return Fluent(self.ref_key, BoolType())

    @property
    def ref_key(self) -> str:
        boat_bank = "left" if self.boat_on_left else "right"
        return f"at_p{self.priests_left}_c{self.cannibals_left}_{boat_bank}"


def define_safe_states(n: int) -> tuple[Problem, dict[State, Fluent]]:
    problem = Problem(f"priests_and_cannibals_{n}")
    states: dict[State, Fluent] = {}

    for priests_left in range(n + 1):
        for cannibals_left in range(n + 1):
            for boat_on_left in (True, False):
                state = State(
                    priests_left,
                    cannibals_left,
                    boat_on_left,
                )
                if not state.is_safe(n):
                    continue

                problem.add_fluent(state.fluent, default_initial_value=False)
                states[state] = state.fluent

    initial_state = states[State(n, n, True)]
    problem.set_initial_value(initial_state, True)
    return problem, states


def print_states(n: int, problem: Problem, states: dict[State, Fluent]) -> None:
    print(f"Safe states for N={n}:")
    for state in states:
        print(f"  {state.ref_key}")

    initial_state = State(n, n, True)
    initial_state_fluent = states[initial_state]
    print(f"\nInitial state: {initial_state.ref_key}")
    print(f"Initial value: {problem.initial_value(initial_state_fluent)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Define explicitly encoded safe priests and cannibals states."
    )
    parser.add_argument(
        "-n",
        "--number",
        type=int,
        default=3,
        help="number of priests and cannibals (default: 3)",
    )
    args = parser.parse_args()
    if args.number < 1:
        parser.error("--number must be at least 1")
    return args


def main() -> None:
    args = parse_args()
    problem, states = define_safe_states(args.number)
    print_states(args.number, problem, states)


if __name__ == "__main__":
    main()
