function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
X = [ones(m,1) X];

Z2 = Theta1*X';

A2=sigmoid(Z2);

A2 =A2';

A2 = [ones(m,1) A2];

A2 = A2';

Z3 = Theta2*A2;

h=sigmoid(Z3);

[b,p] = max(h,[],1);
p=p';
fprintf('\nTraining Set Accuracy: %f\n', mean(double(p == y)) * 100);

  h=h';


k = num_labels;

eyem = eye(num_labels);
v = eyem(y,:);
y=v;

for M = 1:m
for K = 1:k
    
    J = J + (   (1/m) * sum((-y( M,K ) .* log(h( M,K ))) + (-(1-y( M,K )) .* log(1-h( M,K ))))   );
end
end

A = size(Theta1,1);
B = size(Theta1,2);
C = size(Theta2,1);
D = size(Theta2,2);

THE1 =0; THE2 =0;

for j = 1:A
    for K = 2:B
        THE1 = THE1 + (Theta1(j,K))^2;
    end
end

for j = 1:C
    for K = 2:D
        THE2 = THE2 + (Theta2(j,K))^2;
    end
end

THE = (lambda/(2*m)) * ( THE1 + THE2 );

J = J + THE;
%fwd prop done
A1 = X;
del3 = h' - y';
A2 = sigmoid(Z2);
Thet2 = Theta2;
Theta2 = Theta2(:,2:end);
del2 = Theta2'*del3.*sigmoidGradient(Z2);
%del2 = del2(2:end,:);
Del1 =0; Del2=0;
Del1 = Del1 + del2*A1;
A2 = [ones(1,m);sigmoid(Z2)];
Del2 = Del2 + del3*A2';

Theta1_grad = Del1./m;
Theta2_grad = Del2./m;
size(Theta1_grad)
size(Theta2_grad)

for i = 1:size(Theta1_grad,1)
    for j = 2:size(Theta1_grad,2)
        Theta1_grad(i,j) = Theta1_grad(i,j) + (lambda/m)*Theta1(i,j);
    end
end


for i = 1:size(Theta2_grad,1)
    for j = 2:size(Theta2_grad,2)
        Theta2_grad(i,j) = Theta2_grad(i,j) + (lambda/m)*Thet2(i,j);
    end
end
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
