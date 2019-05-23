
clear

global gamma Pr mu0 R Cv Cp K LHORI Twall T0 M0 u0 p0 x y Re dx dy jmax kmax

LHORI = 0.00001; % plate length
Twall = 288.16; % wall temperature
adiabaticWall = true; % if false, enforce Twall temp at surface




gamma = 1.4;
Pr = 0.71; % Prandtl number
mu0 = 1.7894E-5; % dynamic viscosity of air
R = 287; % specific gas constant
Cv = 0.7171 * 1000; % specific heat capacity of air - I assume we want these in J not kJ
Cp = 1.006 * 1000; % specifc heat capacity of air - I assume we want these in J not kJ

K = 0.2; % Courant number -- acts as a fudge factor(?) in equation 10.16


%% define initial values
a0 = 340.28;
p0 = 101325.0;
T0 = 288.16;
M0 = 4.0;
u0 = M0*a0;
v0 = 0;
rho0 = p0 / (R * T0);
e0 = T0 * Cv;

% [rho0, a0, mu0, Et0] = derivedParameters(u0, v0, p0, T0); % this works for single values or matrices

% and get the reynolds number now that we've defined u
Re = rho0 * u0 * LHORI / mu0;



%% set up grid

xmax = LHORI;
ymax = 5 * BoundaryLayerThickness(LHORI);

jmax = 70;
kmax = 70;
% dx = (jmax+1)/LHORI;
% dy = dx; % just make it a square grid?
%
% xVals = 0 : dx : (jmax-1)*dx;
% yVals = 0: dy : (kmax-1)*dy;

xVals = linspace(0, xmax, jmax);
yVals = linspace(0, ymax, kmax);
dx = xVals(2) - xVals(1);
dy = yVals(2) - yVals(1);


x = zeros(jmax,kmax);
y = zeros(jmax,kmax);

for k = 1:kmax
    x(:,k) = xVals;
end
for j = 1:jmax
    y(j,:) = yVals;
end

%% set up matrices with initial conditions
u = u0 * ones(size(x));
v = v0 * ones(size(x));
p = p0 * ones(size(x));
T = T0 * ones(size(x));
rho = rho0 * ones(size(x));
e = e0 * ones(size(x));
% and I think we just need to set the surface BC with u=0?
for j=1:jmax
    u(j,1) = 0;
end



Q = zeros(jmax, kmax, 4);
F = zeros(jmax, kmax, 4);
G = zeros(jmax, kmax, 4);

Q_pred = zeros(jmax, kmax, 4); % need to keep both Q and Q_pred, but F and G can be overwritten between stages


%% calculate our initial dt
% lets initially plan to just do this on CPU and return back the primatives
% every time step
for it = 1:10000
    
    it
    
    dt = calc_dt(u, v, p, T);
    
    % =======================================================================
    % LAUNCH KERNEL HERE
    % =======================================================================
    
    %% update Q, F, and G
    % updating F and G is definitely our most expensive totaly independent
    % operation, so we can at least do that on the GPU
    % but if we're careful with syncing our threads I think we can get everyone to stop and wait at the right times
    
    for j = 1:jmax
        for k = 1:kmax
            Q = calc_Q(Q, u, v, p, T, j, k);
        end
    end
    
    % for debugging:
    Q1 = Q(:,:,1);
    Q2 = Q(:,:,2);
    Q3 = Q(:,:,3);
    Q4 = Q(:,:,4);
    
    
    isPredictor = true;
    for j = 1:jmax
        for k = 1:kmax
            [F, G] = calc_FG(F, G, u, v, p, T, isPredictor, j, k);
        end
    end
    
    % for debugging:
    F1 = F(:,:,1);
    F2 = F(:,:,2);
    F3 = F(:,:,3);
    F4 = F(:,:,4);
    G1 = G(:,:,1);
    G2 = G(:,:,2);
    G3 = G(:,:,3);
    G4 = G(:,:,4);
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    %% run Maccormack's predictor (for interior points only)
    % this is for a uniform mesh but we have that other non-uniform mesh Maccormack's
    for j = 1:jmax
        for k = 1:kmax
            
            Q_pred = MaccormackPredictorUniform(Q_pred, Q, F, G, dt, j, k);
            
            % and we can just update the primatives here without waiting for a
            % sync right?
            [rho, u, v, e, p, T] = primativesFromQ(rho, u, v, e, p, T, Q_pred, j, k);
            
        end
    end
    
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    % actually with the extrapolation we need all the primatives to finish
    % calculating before applying BC
    
    % note that the vast majority of threads will be sitting idle here, but
    % can't think of any better options, and it's a super quick operation
    for j = 1:jmax
        for k = 1:kmax
            [u, v, p, T] = enforceBC_nonSurface(u, v, p, T, j, k);
        end
    end
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    % and if the outflow extrapolation overlaps with the surface
    % extrapolation, need to do them in separate stages so they don't step
    % on each-others toes (or really just occur in a random order, which we
    % don't want)
    
    % note that the vast majority of threads will be sitting idle here, but
    % can't think of any better options, and it's a super quick operation
    for j = 1:jmax
        for k = 1:kmax
            [u, v, p, T] = enforceBC_surface(u, v, p, T, j, k, adiabaticWall);
        end
    end
    
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    %% update F and G for Q_pred
    isPredictor = false;
    for j = 1:jmax
        for k = 1:kmax
            [F, G] = calc_FG(F, G, u, v, p, T, isPredictor, j, k);
        end
    end
    
    % for debugging:
    F1 = F(:,:,1);
    F2 = F(:,:,2);
    F3 = F(:,:,3);
    F4 = F(:,:,4);
    G1 = G(:,:,1);
    G2 = G(:,:,2);
    G3 = G(:,:,3);
    G4 = G(:,:,4);
    
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    %% and then run Maccormack's corrector (for interior points only)
    
    for j = 1:jmax
        for k = 1:kmax
            Q = MaccormackCorrectorUniform(Q, Q_pred, F, G, dt, j, k);
            
            % and we can just update the primatives here without waiting for a
            % sync right?
            [rho, u, v, e, p, T] = primativesFromQ(rho, u, v, e, p, T, Q, j, k);
            
        end
    end
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    % actually with the extrapolation we need all the primatives to finish
    % calculating before applying BC
    
    % note that the vast majority of threads will be sitting idle here, but
    % can't think of any better options, and it's a super quick operation
    for j = 1:jmax
        for k = 1:kmax
            [u, v, p, T] = enforceBC_nonSurface(u, v, p, T, j, k);
        end
    end
    
    % =======================================================================
    % MUST SYNC THREADS HERE (but I think we can continue within same kernel)
    % =======================================================================
    
    % and if the outflow extrapolation overlaps with the surface
    % extrapolation, need to do them in separate stages so they don't step
    % on each-others toes (or really just occur in a random order, which we
    % don't want)
    
    % note that the vast majority of threads will be sitting idle here, but
    % can't think of any better options, and it's a super quick operation
    for j = 1:jmax
        for k = 1:kmax
            [u, v, p, T] = enforceBC_surface(u, v, p, T, j, k, adiabaticWall);
        end
    end
    
    
    % and that's it right?
    % copy back u, v, p, T
    
%     figure(1)
%     plot(p(:,1) / p0)
%     
%     figure(2)
%     plot(T(:,1) / T0)
    
    
end






