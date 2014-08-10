function [J, grad] = linearRegCostFunction(X, y, theta, lambda)
%LINEARREGCOSTFUNCTION Compute cost and gradient for regularized linear 
%regression with multiple variables
%   [J, grad] = LINEARREGCOSTFUNCTION(X, y, theta, lambda) computes the 
%   cost of using theta as the parameter for linear regression to fit the 
%   data points in X and y. Returns the cost in J and the gradient in grad

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost and gradient of regularized linear 
%               regression for a particular choice of theta.
%
%               You should set J to the cost and grad to the gradient.
%

J_without_reg_term = (0.5/m) * sum((X*theta - y).^2);  % Cost without the regularization term
J_reg_term = (lambda/(2*m)) * sum(theta(2:end).^2); % Regularization term

J = J_without_reg_term + J_reg_term;

h = X * theta;
error = h - y;
%grad_without_reg_term = (1/m) * error' * X;  % Gradient for all terms
grad_without_reg_term = (1/m) * X' * error;  % Gradient for all terms
grad_reg_term = (lambda/m) * ([0 ; theta(2:end)]); % Regularization for Gradient

%grad = grad_without_reg_term' + grad_reg_term; % Grad is a vector that includes the regularization term
grad = grad_without_reg_term + grad_reg_term; % Grad is a vector that includes the regularization term



% =========================================================================

grad = grad(:);

end
