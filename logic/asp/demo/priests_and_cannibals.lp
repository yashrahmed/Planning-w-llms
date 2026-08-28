#const n = 3.
#const h = 11.

count(0..n).
side(left; right).
step(0..h-1).

% A bank with no priests is safe regardless of its cannibal count.
bank_safe(0, C) :-
    count(C).

% Otherwise, the priests must not be outnumbered by cannibals.
bank_safe(P, C) :-
    count(P),
    count(C),
    P > 0,
    P >= C.

% PL and CL are the priest and cannibal counts on the left bank.
% The right-bank counts are inferred from the total population n.
safe(PL, CL) :-
    count(PL),
    count(CL),
    PR = n - PL,
    CR = n - CL,
    bank_safe(PL, CL),
    bank_safe(PR, CR).

% Everyone and the boat begin on the left bank.
at(n, n, left, 0).

% When the boat is on the left, choose exactly one safe next state on the right.
1 {
    at(PL2, CL2, right, T + 1) :
        safe(PL2, CL2),
        PL2 <= PL,
        CL2 <= CL,
        (PL - PL2) + (CL - CL2) >= 1,
        (PL - PL2) + (CL - CL2) <= 2
} 1 :-
    at(PL, CL, left, T),
    step(T).

% When the boat is on the right, choose exactly one safe next state on the left.
1 {
    at(PL2, CL2, left, T + 1) :
        safe(PL2, CL2),
        PL2 >= PL,
        CL2 >= CL,
        (PL2 - PL) + (CL2 - CL) >= 1,
        (PL2 - PL) + (CL2 - CL) <= 2
} 1 :-
    at(PL, CL, right, T),
    step(T).

% At the selected horizon, everyone and the boat must be on the right bank.
:- not at(0, 0, right, h).

#show at/4.
