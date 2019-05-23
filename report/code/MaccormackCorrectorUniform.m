function Q = MaccormackCorrectorUniform(Q, Q_pred, F_pred, G_pred, dt, j, k)

% kernel function

global dx dy jmax kmax

% DO MACCORMACKS FOR FOR INTERIOR POINTS ONLY
% if an edge point, just do nothing
if j == 1 || k == 1 || j == jmax || k == kmax
    return;
end


% have each thread calculate all 4 dimensions at a single loc
for dim = 1:4
    
    flux = (F_pred(j,k,dim) - F_pred(j-1,k,dim))/dx + (G_pred(j,k,dim) - G_pred(j,k-1,dim))/dy;
    Q(j,k,dim) = 0.5*( Q(j,k,dim) + Q_pred(j,k,dim) - dt * flux);
    
end


end