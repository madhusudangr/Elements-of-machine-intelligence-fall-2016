close all;
clear;
clc;

filter_Order = [4,8,15,25,30]; % index_1
lambda = [0.001,0.020,0.04,0.08,0.1]; % index_2
load('HW2/training.mat');
data = training;
for index_1 =1:5
    for index_2 =1:5
        W(index_1,index_2) = calc_W(filter_Order(index_1),lambda(index_2),data);
        MSE(index_1,index_2) = validation(filter_Order(index_1),lambda(index_2),W(index_1,index_2));
    end
end

lambda_plot = zeros(25,1); % index_3
filter_order_plot = zeros(25,1); % index_3
MSE_plot = zeros(25,1); % index_3
index_3 =1;
for index_1 =1:5
    for index_2 =1:5
        filter_order_plot(index_3,1) = filter_Order(index_1);
        lambda_plot(index_3,1) = lambda(index_2);
        MSE_plot(index_3,1) = MSE(index_1,index_2); 
        index_3 = index_3 + 1;
    end
end

figure
surf(lambda,filter_Order,MSE);
title('Performance Plot/ Performance Surface');
xlabel('lambda');
ylabel('Filter Order');
zlabel('MSE');

min_ERROR = min(min(MSE));
index = find(MSE == min_ERROR);

%so we get min error at lambda = 0.001 and filter order at 4 where W = 0.996048629942459
% so using that to predict for the test data
load('HW2/test.mat');
X = test;
[MSE, Y, Y_bar] = test1(X, 4  ,0.001,0.996048629942459);
[MSE1, Y1, Y_bar1] = test1(X, 8  ,0.001,0.996048629942459);
[MSE2, Y2, Y_bar2] = test1(X, 30  ,0.001,0.996048629942459);

figure
title('Error & Input vs Time for Order 4');
xlabel('Time');
plot((1:1:1000),MSE);
ylabel('Error');
yyaxis right
plot((1:1:1000),X);
ylabel('X-input');

figure
title('Error & Input vs Time for Order 8');
xlabel('Time');
plot((1:1:1000),MSE1);
ylabel('Error');
yyaxis right
plot((1:1:1000),X);
ylabel('X-input');
figure
title('Error & Input vs Time for Order 30');
xlabel('Time');
plot((1:1:1000),MSE2);
ylabel('Error');
yyaxis right
plot((1:1:1000),X);
ylabel('X-input');


%third question 
load('HW2/testnoisy.mat');
X = testnoisy;
[MSE, Y, Y_bar] = test1(X, 4  ,0.001,0.996048629942459);
[MSE1, Y1, Y_bar1] = test1(X, 8  ,0.001,0.996048629942459);
[MSE2, Y2, Y_bar2] = test1(X, 30  ,0.001,0.996048629942459);



