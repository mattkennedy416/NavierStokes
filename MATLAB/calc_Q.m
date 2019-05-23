function Q = calc_Q(Q, u, v, p, T, j, k)

% kernel function

global R Cv

rho_val = p(j,k) ./ (R * T(j,k));
e_val = Cv * T(j,k); % energy of air based on temp
Et = rho_val .* (e_val + (u(j,k)^2 + v(j,k)^2)/2); % total energy

Q(j,k,1) = rho_val;
Q(j,k,2) = rho_val * u(j,k);
Q(j,k,3) = rho_val * v(j,k);
Q(j,k,4) = Et;

end