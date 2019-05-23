function [F, G] = calc_FG(F, G, u, v, p, T, isPredictor, j, k)

% kernel function

% lets also track mu from this function since I think the only other place
% that needs it is the time step update


% update F and G for location (j,k)
% inputs are assumed to be entire matrices

global R Cv T0 mu0

rho_val = p(j,k) / (R * T(j,k));
e_val = Cv * T(j,k); % energy of air based on temp
Et_val = rho_val * (e_val + (u(j,k)^2 + v(j,k)^2)/2); % total energy

% Sutherlands Law:
mu_val = mu0*(T(j,k) / T0).^1.5 * (T0 + 110)./(T(j,k) + 110);

[qx, qy] = heatFluxParameters(T, mu_val, isPredictor, j, k);
[txx, tyy, txy_F, txy_G] = shearParameters(u, v, mu_val, isPredictor, j, k);

F(j,k,1) = rho_val * u(j,k);
F(j,k,2) = rho_val * u(j,k)^2 + p(j,k) - txx;
F(j,k,3) = rho_val * u(j,k) * v(j,k) - txy_F;
F(j,k,4) = (Et_val + p(j,k)) * u(j,k) - u(j,k) * txx - v(j,k) * txy_F + qx;

G(j,k,1) = rho_val * v(j,k);
G(j,k,2) = rho_val * u(j,k) * v(j,k) - txy_G;
G(j,k,3) = rho_val * v(j,k)^2 + p(j,k) - tyy;
G(j,k,4) = (Et_val + p(j,k)) * v(j,k) - u(j,k) * txy_G - v(j,k) * tyy + qy;



% for debugging:
F1 = F(:,:,1);
F2 = F(:,:,2);
F3 = F(:,:,3);
F4 = F(:,:,4);
G1 = G(:,:,1);
G2 = G(:,:,2);
G3 = G(:,:,3);
G4 = G(:,:,4);


end