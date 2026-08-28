# Explicit priests and cannibals states

This first step defines only the safe states for the generalized priests and
cannibals problem. It does not define actions, goals, or invoke a planner yet.

Each complete configuration is represented by one Boolean fluent. For example,
`at_p3_c1_right` means that three priests and one cannibal are on the left bank
and the boat is on the right bank. The populations on the right bank are
inferred from `N`.

An immutable `State` class encapsulates the left-bank counts and boat location.
It derives the right-bank counts, checks both banks for safety, and generates
the corresponding fluent name.

Only configurations where neither bank has more cannibals than priests—unless
that bank has no priests—are defined.

Run it with the default `N=3`:

```sh
uv run python demo.py
```

Choose a different value of `N`:

```sh
uv run python demo.py --number 5
```
