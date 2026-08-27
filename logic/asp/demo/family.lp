% Clingo program-part names use identifier syntax (underscores, not hyphens).
#program yjd_custom_program.

% Facts: parent(Parent, Child).
parent(alice, bob).
parent(bob, carol).
parent(carol, dave).
parent(alice, erin).
parent(erin, frank).

% A grandparent is the parent of a parent.
grandparent(X, Z) :- parent(X, Y), parent(Y, Z).

% Every parent is an ancestor.
ancestor(X, Y) :- parent(X, Y).

% Ancestorship is transitive through a parent.
ancestor(X, Z) :- parent(X, Y), ancestor(Y, Z).

% Keep the answer set focused on the two derived relations.
#show grandparent/2.
#show ancestor/2.
