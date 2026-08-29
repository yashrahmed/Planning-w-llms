

value(0..10).

fibo(0, 0).
fibo(1, 1).

fibo(N, Res) :-
    value(N),
    N >= 2,
    X = N - 1,
    Y = N - 2,
    fibo(X, F1),
    fibo(Y, F2),
    Res = F1 + F2.

#show fibo/2.