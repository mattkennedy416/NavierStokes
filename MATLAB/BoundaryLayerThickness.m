function blt = BoundaryLayerThickness(LHORI)
% Blasius calculation

global Re

blt = 5*LHORI / sqrt(Re);

end