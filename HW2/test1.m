function [MSE,Y,Y_bar ] = test1(X,filterOrder, lambda, W )

Y = zeros(size(X,1),1);
for index=2:size(X,1)
    if(index-filterOrder>0)
        Y(index,1) = X(index-filterOrder);
    else
        Y(index,1) = 0;
    end
end
for index=2:size(X,1)
    if(index-filterOrder>0)
        Y_bar(index,1) = W * X(index - filterOrder,1);
    else
        Y_bar(index,1) = 0;
    end
end

%MSE = mean((Y - Y_bar).^2);

for index=1:size(Y,1)
    MSE(index,1) = mean( (Y(index) - Y_bar(index))^2  );
end

meanMSE = mean((Y-Y_bar).^2);
disp(sprintf(' For test data ---'));
disp(sprintf('filter order: %d, lambda: %d, W: %d, MSE: %d',filterOrder,lambda,W,meanMSE));
end



