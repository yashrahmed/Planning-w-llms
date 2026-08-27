from pathlib import Path

import clingo


PROGRAM = Path(__file__).with_name("family.lp")
PROGRAM_PART = "yjd_custom_program"


def format_relation(symbol: clingo.Symbol) -> str:
    left, right = symbol.arguments
    return f"{left} -> {right}"


def print_model(model: clingo.Model) -> None:
    shown = model.symbols(shown=True)
    print(shown)
    print('________')

    for fact in shown:
        print(fact.arguments[0], fact.arguments[1], fact.name)

    # for relation, heading in (
    #     ("grandparent", "Grandparents"),
    #     ("ancestor", "Ancestors"),
    # ):
    #     print(relation)
    #     print(heading)
    #     print('___________')
        # matches = sorted(
        #     (symbol for symbol in shown if symbol.name == relation),
        #     key=str,
        # )
        # print(f"{heading}:")
        # for symbol in matches:
        #     print(f"  {format_relation(symbol)}")


def main() -> None:
    control = clingo.Control()
    control.load(str(PROGRAM))
    control.ground([(PROGRAM_PART, [])])
    result = control.solve(on_model=print_model)

    if not result.satisfiable:
        raise SystemExit("The ASP program has no answer set.")


if __name__ == "__main__":
    main()
