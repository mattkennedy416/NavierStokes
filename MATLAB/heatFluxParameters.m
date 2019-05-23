function [qx, qy] = heatFluxParameters(T, mu_val, isPredictor, j, k)

% calculate the heat flux parameters for location (j,k)
% inputs T and mu are entire matrices

global Cp Pr dx dy jmax kmax

k_cond = mu_val * Cp / Pr; % conductivity parameter using mu from Sutherlands Law


if isPredictor % scheme is forward, make this backward
    
    if j > 1
        dTdx = (T(j,k) - T(j-1,k))/dx;
    else
        dTdx = (T(2,k) - T(1,k))/dx;
    end
    
    if k > 1       
        dTdy = (T(j,k) - T(j,k-1))/dy;
    else
        dTdy = (T(j,2) - T(j,1))/dy;
    end
    
else
    
    if j < jmax
        dTdx = (T(j+1,k) - T(j,k))/dx;
    else
        dTdx = (T(jmax,k) - T(jmax-1,k))/dx;
    end
    
    if k < kmax    
        dTdy = (T(j,k+1) - T(j,k))/dy;
    else
        dTdy = (T(j,kmax) - T(j,kmax-1))/dy;
    end
    
end

qx = -k_cond * dTdx;
qy = -k_cond * dTdy;



end