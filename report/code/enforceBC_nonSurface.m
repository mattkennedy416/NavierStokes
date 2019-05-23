

function [u, v, p, T] = enforceBC_nonSurface(u, v, p, T, j, k)

% need to first establish all the boundary conditions at the non-surface
% values, and then go back and do the surface boundary conditions

% this is really only needed if the surface goes all the way to the outflow
% so that the last surface point can be interpolated with updated values

global p0 T0 u0 kmax jmax


if j == 1 && k == 1  % leading edge
    u(j,k) = 0;
    v(j,k) = 0;
    p(j,k) = p0;
    T(j,k) = T0;
    
elseif j == 1 || k == kmax  % inflow from upstream OR upper boundary
    u(j,k) = u0;
    v(j,k) = 0;
    p(j,k) = p0;
    T(j,k) = T0;
    
elseif j == jmax % outflow
    % extrapolate from interior values
%     u(j,k) = 2*u(j,k-1) - u(j,k-2);
%     v(j,k) = 2*v(j,k-1) - v(j,k-2);
%     p(j,k) = 2*p(j,k-1) - p(j,k-2);
%     T(j,k) = 2*p(j,k-1) - p(j,k-2);
    u(j,k) = 2*u(j-1,k) - u(j-2,k);
    v(j,k) = 2*v(j-1,k) - v(j-2,k);
    p(j,k) = 2*p(j-1,k) - p(j-2,k);
    T(j,k) = 2*T(j-1,k) - T(j-2,k);
end
   

end


