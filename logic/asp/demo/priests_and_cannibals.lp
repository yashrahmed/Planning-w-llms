#const n = 3.
#const h = 110.

count(0..n).
boat_side(boat_left; boat_right).
step(0..h-1).


% PL and CL are the priest and cannibal counts on the left bank.
% The boat_right-bank counts are inferred from the total population n.
safe(PL, CL) :-
    count(PL),
    count(CL),
    PR = n - PL,
    CR = n - CL,

    count(PL),
    count(CL),
    PL >= CL : PL > 0;

    count(PR),
    count(CR),
    PR >= CR : PR > 0.

% Everyone and the boat begin on the left bank.
state(n, n, boat_left, 0).

% When the boat is on the left, choose exactly one safe next state on the boat_right.
1 {
    state(PL2, CL2, boat_right, T + 1) :
        safe(PL2, CL2),
        PL2 <= PL,
        CL2 <= CL,
        (PL - PL2) + (CL - CL2) >= 1,
        (PL - PL2) + (CL - CL2) <= 2
} 1 :-
    state(PL, CL, boat_left, T),
    step(T).

% When the boat is on the boat_right, choose exactly one safe next state on the left.
1 {
    state(PL2, CL2, boat_left, T + 1) : % ':' read as 'such that'
        safe(PL2, CL2),
        PL2 >= PL,
        CL2 >= CL,
        (PL2 - PL) + (CL2 - CL) >= 1,
        (PL2 - PL) + (CL2 - CL) <= 2
} 1 :-
    state(PL, CL, boat_right, T),
    step(T).

% At the selected horizon, everyone and the boat must be on the boat_right bank.
:- not state(0, 0, boat_right, h).

#show state/4.
