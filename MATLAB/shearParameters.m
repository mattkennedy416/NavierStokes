function [txx, tyy, txy_F, txy_G] = shearParameters(u, v, mu, isPredictor, j, k)

% calculate shear for a single location (j,k)
% inputs are assumed to be entire matrices

global dx dy jmax kmax


%% calculate the forward or backward differenced versions
if isPredictor
    % want opposite direction from scheme step differencing
    % scheme is forward, make this backward
    
    if j > 1
        dvdx_FB = (v(j,k) - v(j-1,k))/dx;
        dudx_FB = (u(j,k) - u(j-1,k))/dx;     
    else
        dvdx_FB = (v(2,k) - v(1,k))/dx; % except first point forward
        dudx_FB = (u(2,k) - u(1,k))/dx; % except first point forward
    end
    
    if k > 1
        dudy_FB = (u(j,k) - u(j,k-1))/dy;
        dvdy_FB = (v(j,k) - v(j,k-1))/dy;
    else
        dudy_FB = (u(j,2) - u(j,1))/dy; % except first point forward
        dvdy_FB = (v(j,2) - v(j,1))/dy; % except first point forward
    end
    
else
    
    % scheme is backward, make this forward
    
    if j < jmax
        dvdx_FB = (v(j+1,k) - v(j,k))/dx;
        dudx_FB = (u(j+1,k) - u(j,k))/dx;
    else
        dvdx_FB = (v(j,k) - v(j-1,k))/dx; % except jmax backward
        dudx_FB = (u(j,k) - u(j-1,k))/dx; % except jmax backward
    end
        
    if k < kmax
        dudy_FB = (u(j,k+1) - u(j,k))/dy;
        dvdy_FB = (v(j,k+1) - v(j,k))/dy;
    else
        dudy_FB = (u(j,kmax) - u(j,kmax-1))/dy; % except kmax backward
        dvdy_FB = (v(j,kmax) - v(j,kmax-1))/dy; % except kmax backward
    end
    
end

%% and then we want centeral differenced versions

if j == 1
    dvdx_C = (v(2,k) - v(1,k))/dx;
    dudx_C = (u(2,k) - u(1,k))/dx;
elseif j == jmax
    dvdx_C = (v(jmax,k) - v(jmax-1,k))/dx;
    dudx_C = (u(jmax,k) - u(jmax-1,k))/dx;
else
    dvdx_C = (v(j+1,k) - v(j-1,k))/(2*dx);
    dudx_C = (u(j+1,k) - u(j-1,k))/(2*dx);
end


if k == 1
    dudy_C = (u(j,2) - u(j,1))/dy;
    dvdy_C = (v(j,2) - v(j,1))/dy;
elseif k == kmax
    dudy_C = (u(j,kmax) - u(j,kmax-1))/dy;
    dvdy_C = (v(j,kmax) - v(j,kmax-1))/dy;
else
    dudy_C = (u(j,k+1) - u(j,k-1))/(2*dy);
    dvdy_C = (v(j,k+1) - v(j,k-1))/(2*dy);
end


% these come from page 65 and 66 in Anderson

lambda = -(2/3) * mu; % second viscosity coefficient estimated by Stokes

% use the forward/backward du/dx and central dv/dy for both F and G
txx = lambda * ( dudx_FB + dvdy_C ) + 2 * mu * dudx_FB;

% use the forward/backward dv/dy and central du/dx for both F and G
tyy = lambda * ( dudx_C + dvdy_FB ) + 2 * mu * dvdy_FB; 

txy_F = mu * ( dvdx_FB + dudy_C );
txy_G = mu * ( dvdx_C + dudy_FB );



end









