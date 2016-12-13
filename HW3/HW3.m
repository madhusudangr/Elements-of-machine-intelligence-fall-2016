clc
clear
close all

load('HW3/validate.mat');
load('HW3/training.mat');
load('HW3/test.mat');
N = length(training);
Nv= length(validate);
M_count =1; mu_count=1;
M_list = (1 :1: 100);
mu_min = 0.001;
mu_max = 10;
bin = ((mu_max-mu_min)/99);
mu_list = (mu_min :bin:mu_max);


%for validation
for M = M_list
    for mu = mu_list
        %WT validate(Nv?i : ?1 : Nv?j?M +1)?validate(Nv?j+1)
        for i=1:Nv-M
            X = validate(Nv-i: -1 :Nv-i-M+1);
            y = validate(Nv-i+1);
            W = zeros(M,1);
            %W = calculate_gradient(X,y,W,mu);
            for index=1:size(W)
                gradient = (X'* W - y);
                tempW(index) = W(index) + 2*(mu) * gradient' * X(index);
            end
            W = tempW';
            c1 = 1/(2*M);
            c2 = (X' * W - y)';
            c3 = (X' * W - y);
            CostValidate(M_count,mu_count,i) = c1 * c2 * c3;
        end
%         subplot(M_count*mu_count,M_count,mu_count)
%         plot((1:1:N-M),Cost(M_count,mu_count,Cost(M_count,mu_count,:)~=0));
        MSE_Validate(M_count,mu_count) = (CostValidate(M_count,mu_count,Nv-M));
        mu_count = mu_count+1;
    end
    mu_count = 1;         
    M_count = M_count+1;
end    

figure
surf(M_list,mu_list,MSE_Validate);
saveas(gcf,'surf_validate');



% M_count =1; mu_count=1;
% for M = M_list
%     for mu = mu_list
%         %WT validate(Nv?i : ?1 : Nv?j?M +1)?validate(Nv?j+1)
%         for i=1:N-M
%             X = training(N-i: -1 :N-i-M+1);
%             y = training(N-i+1);
%             W = zeros(M,1);
%             %W = calculate_gradient(X,y,W,mu);
%             for index=1:size(W)
%                 gradient = (X'* W - y);
%                 tempW(index) = W(index) + (mu) * gradient' * X(index);
%             end
%             W = tempW';
%             c1 = 1/(2*M);
%             c2 = (X' * W - y)';
%             c3 = (X' * W - y);
%             Cost(M_count,mu_count,i) = c1 * c2 * c3;
%         end
% %         subplot(M_count*mu_count,M_count,mu_count)
% %         plot((1:1:N-M),Cost(M_count,mu_count,Cost(M_count,mu_count,:)~=0));
%          MSE(M_count,mu_count) = (Cost(M_count,mu_count,N-M));
%         mu_count = mu_count+1;
%     end
%     mu_count = 1;         
%     M_count = M_count+1;
% end

% M = (1 :1: 70);
% mu = (0.01 :bin:10);
% figure
% surf(M_list,mu_list,MSE);
% saveas(gcf,'surf');

[v,d]= eig(training'*training);
theoretical_mu = 1/trace(d);

% M_count =1; mu_count=1;
% for M = M_list
%     for mu = mu_list
%         MSE_min(M_count,mu_count) = min(Cost(M_count,mu_count,:));
%         mu_count = mu_count+1;
%     end
%     mu_count = 1;         
%     M_count = M_count+1;
% end
% 
% figure
% surf(M_list,mu_list,MSE_min);
% saveas(gcf,'surf_MSE_min');
