
person(alice).
person(bob).
person(charlie).
person(dave).
person(edith).
person(frank).
person(jenny).
person(haley).


age(alice, 10).
age(bob, 11).
age(charlie, 15).
age(dave, 40).
age(edith, 30).
age(frank, 37).
age(jenny, 19).
age(haley, 10).

% Terms on numbers can be used as a constraint if a solver explores a search space for that numeric variable.
is_adult(P) :- person(P),
               age(P, Age), % Age binds to values from available facts. The solves does not calculate possible Age values and therefore do not need a constraint.
               Age >= 18.

#show is_adult_in_3/1.