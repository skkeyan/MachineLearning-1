function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


% Refer Forum thread - https://class.coursera.org/ml-006/forum/thread?thread_id=1098

term_1 = X * theta;
term_2 = sigmoid(term_1);  % Calculate h by using the sigmoid(z) function
term_3 = log(term_2);
term_4 = log(1-term_2);
term_5 = -1 * y' * term_3;
term_6 = (1-y') * term_4;

J_aterm = (1/m) * (term_5 - term_6);   % Cost function without the regularization term
J_bterm = (lambda/(2*m)) * sum(theta(2:end).^2); % Regularization term
J = J_aterm + J_bterm;   % Total Cost

% Calculating the gradient

h = sigmoid(X * theta);  
error = h - y;
grad_aterm = (1/m) * error' * X;  % Gradient for all terms
grad = grad_aterm' + (lambda/m) * ([0 ; theta(2:end)]); % Grad is a vector that includes the regularization term

% =============================================================

end
