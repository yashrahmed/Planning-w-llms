% The world contains one web client, one customer, and one service.
web_client(w).
customer(customer_a).
service(customer_service).
service_has_data(customer_service, customer_a).

% Initially, the client is missing the customer's data.
client_data_state(w, customer_a, missing, 0).

% At step 0, the client calls the service for that customer.
occurs(call_service(customer_service, customer_a), 0).

valid_call(Service, Customer) :-
    service(Service),
    customer(Customer),
    service_has_data(Service, Customer).

% Calls to nonexistent or unsuitable services are not allowed.
:- occurs(call_service(Service, Customer), _), not valid_call(Service, Customer).

% A successful call changes the client's state at the next step.
client_data_state(W, Customer, available, T + 1) :-
    client_data_state(W, Customer, missing, T),
    occurs(call_service(Service, Customer), T),
    web_client(W).

#show occurs/2.
#show client_data_state/4.
