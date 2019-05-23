function dt = calc_dt(u, v, p, T)

% for finding the max/min I'm not sure the best way to do this on the GPU
% lets just plan on doing it on the CPU initially

global gamma Pr dx dy K jmax kmax R mu0 T0

%dv = max(max( (4/3).*mu.*(gamma*mu/Pr)./rho ));

dv = -inf;
for j = 1:jmax
    for k = 1:kmax
        
        rho_val = p(j,k) / (R * T(j,k));
        mu_val = mu0 * (T(j,k) / T0)^1.5 * (T0 + 110)./(T(j,k) + 110);
        
        val = (4/3) * mu_val * (gamma*mu_val/Pr) / rho_val; % find the max of this
        if val > dv
            dv = val;
        end
    end
end




spaceUnit = sqrt(1/dx^2 + 1/dy^2);

% dt = min(min( K * dt_cfl )); 
dt = inf;
for j = 1:jmax
    for k = 1:kmax
        
        rho_val = p(j,k) / (R * T(j,k));
        
        term1 = abs(u(j,k)) / dx;
        term2 = abs(v(j,k)) / dy;
        term3 = sqrt(gamma * p(j,k) / rho_val) * spaceUnit;
        term4 = 2 * dv * spaceUnit*spaceUnit;

        dt_cfl = 1./(term1 + term2 + term3 + term4);
        
        % multiply by Courant number as a "fudge factor" and take the min
        if K*dt_cfl < dt
            dt = K*dt_cfl;
        end
        
    end
end



end