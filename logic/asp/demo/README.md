# ASP demos

## Grandparent and ancestor

This example keeps the family facts and logical clauses in `family.lp`, inside
the custom `yjd_custom_program` program section. The Python launcher explicitly
grounds that section, solves the program, and prints the derived relations.

Run it from this directory:

```sh
uv run python demo.py
```

## Priests and cannibals

`priests_and_cannibals.lp` models each safe configuration as an explicit
`at(PriestsLeft, CannibalsLeft, BoatSide, Step)` atom. It selects the next safe
state directly, without a separate transition predicate.

Solve the default `N=3` problem at its shortest horizon of 11 crossings:

```sh
uv run python -m clingo priests_and_cannibals.lp -c n=3 -c h=11 1
```
