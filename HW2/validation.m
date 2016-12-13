function [ MSE ] = validation( filterOrder, lambda ,W )

load('HW2/validate.mat');
X = validate;
Y = zeros(size(validate,1),1);
for index=2:size(validate,1)
    if(index-filterOrder>0)
        Y(index,1) = X(index-filterOrder);
    else
        Y(index,1) = 0;
    end
end

for index=2:size(validate,1)
    if(index-filterOrder>0)
        Y_bar(index,1) = W * X(index - filterOrder,1);
    else
        Y_bar(index,1) = 0;
    end
end

MSE = mean((Y - Y_bar).^2);
disp(sprintf('filter order: %d, lambda: %d, W: %d, MSE: %d',filterOrder,lambda,W,MSE));


end

