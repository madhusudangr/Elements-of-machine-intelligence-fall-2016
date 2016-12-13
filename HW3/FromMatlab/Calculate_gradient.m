function [W] = Calculate_gradient(X, y, W, mu )

for i=1:size(W)
    gradient = (X * W - y);
    tempW(i) = W + (mu /NormalizeValue) * gradient' * X(:,i);
end
W = tempW;

end