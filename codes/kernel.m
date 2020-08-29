
%polynomial kernel with degree 2 which computes phi(x)*phi(y)'
%here, x and y are row vectors
function k = kernel(x,y)
    k = (1 + x*y').^2;
end
