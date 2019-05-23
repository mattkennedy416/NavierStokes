
function [u, v, p, T] = enforceBC_surface(u, v, p, T, j, k, adiabaticWall)

% need to first establish all the boundary conditions at the non-surface
% values, and then go back and do the surface boundary conditions

% this is really only needed if the surface goes all the way to the outflow
% so that the last surface point can be interpolated with updated values

global Twall

if k == 1 && j > 0
    u(j,k) = 0;
    v(j,k) = 0;
    p(j,k) = 2*p(j,k+1) - p(j,k+2);
    
    if adiabaticWall
        T(j,k) = T(j,k+1);
    else
        T(j,k) = Twall;
    end
end





