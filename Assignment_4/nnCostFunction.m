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

y_matrix = eye(num_labels)(y,:);  % Gives a 5000 * 10 matrix

% Add ones to the X data matrix
a1 = [ones(m, 1) X];
z2 = a1 * Theta1'; % a1 is a 5000 * 401 matrix and Theta1 is a 25 * 401 matrix. z2 becomes 5000 * 25 matrix
a2 = sigmoid(z2);  % a2 is 5000 * 25 matrix    

% Adding 1s to a2 layer
a2 = [ones(m,1) a2]; % a2 is 5000 * 26 matrix

z3 = a2 * Theta2'; % a2 is a 5000 * 26 matrix and Theta2 is a 10*26 matrix. z3 becomes 5000 * 10 matrix  
a3 = sigmoid(z3);  % a3 is 5000 * 10

% Computing cost
term1 = log(a3);  % term1 is 5000 * 10 matrix
term2 = log(1-a3); % term2 is 5000 * 10 matrix

J_first_part = sum(sum(-1 * y_matrix .* term1));
J_second_part = sum(sum((1-y_matrix) .* term2));

%fprintf('Size of J first part = %f\n', J_first_part);
%fprintf('Size of J second part = %f\n', J_second_part);

J = (1/m) * (J_first_part - J_second_part); % Unregularized Cost Function

Theta1_without_bias = Theta1(:,2:end);
Theta2_without_bias = Theta2(:,2:end);

reg_term = (lambda/(2*m)) * (sum(sum(Theta1_without_bias.^2)) + sum(sum(Theta2_without_bias.^2)));
% fprintf('Reg Term = %f\n', reg_term);

J = J + reg_term;    % Regularized Cost Function

% Calculating the gradient

d3 = a3 - y_matrix;   % d3 is a 5000 * 10 matrix
d2 = (d3 * Theta2_without_bias) .* (sigmoidGradient(z2)); % d2 is a 5000 * 25 matrix
%d2 = d2_first_term .* sigmoidGradient(z2);

%D2 = d3' * (a2(:,2:end));   % d3 is a 5000 * 10 matrix. a2 is 5000 * 25 matrix.
%D1 = d2' * (a1(:,2:end));   % d2 is 5000 * 25 matrix. a1 is 5000 * 400 matrix

D2 = d3' * a2;   % d3 is a 5000 * 10 matrix. a2 is 5000 * 25 matrix.
D1 = d2' * a1;   % d2 is 5000 * 25 matrix. a1 is 5000 * 400 matrix

Theta1_grad_unreg = (1/m) * D1;   % Theta1 Gradient without regularization
Theta2_grad_unreg = (1/m) * D2;   % Theta2 Gradient without regularization

Theta1_reg_term = (lambda/m) * Theta1_without_bias;
Theta2_reg_term = (lambda/m) * Theta2_without_bias;

Theta1_grad = [Theta1_grad_unreg(:,1) (Theta1_grad_unreg(:,2:end)+Theta1_reg_term)];  % Theta1 Gradient with regularization
Theta2_grad = [Theta2_grad_unreg(:,1) (Theta2_grad_unreg(:,2:end)+Theta2_reg_term)];  % Theta2 Gradient with regularization


% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
