function Q_pred = MaccormackPredictorUniform(Q_pred, Q, F, G, dt, j, k)

% kernel function

global dx dy jmax kmax

% DO MACCORMACKS FOR FOR INTERIOR POINTS ONLY
% if an edge point, just do nothing
if j == 1 || k == 1 || j == jmax || k == kmax
    return;
end
    

% have each thread calculate all 4 dimensions at a single loc
for dim = 1:4

    Q_pred(j,k,dim) = Q(j,k,dim) - dt * ( (F(j+1,k,dim) - F(j,k,dim))/dx + (G(j,k+1,dim) - G(j,k,dim))/dy );
    
end


end