function [rho, u, v, e, p, T] = primativesFromQ(rho, u, v, e, p, T, Q, j, k)

global Cv R

rho_val = Q(j,k,1); % just to reduce calls to variables in global memory

rho(j,k) = rho_val;
u(j,k) = Q(j,k,2) / rho_val;
v(j,k) = Q(j,k,3) / rho_val;
e(j,k) = Q(j,k,4) / rho_val - 0.5*( u(j,k)^2 + v(j,k)^2 );


% actually want to update p and T here too
T(j,k) = e(j,k) / Cv;
p(j,k) = rho_val * R * T(j,k);


end