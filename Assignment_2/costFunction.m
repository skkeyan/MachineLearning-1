function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

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
%
% Note: grad should have the same dimensions as theta
%

term_1 = X * theta;

term_2 = sigmoid(term_1);

term_3 = log(term_2);
term_4 = log(1-term_2);

term_5 = -1 * y' * term_3;
% fprintf(' %f \n', term_5);
term_6 = (1-y') * term_4;
% fprintf(' %f \n', term_6);

J = (1/m) * (term_5 - term_6); % Cost Function

% Computing the initial gradient value

h = sigmoid(term_1);
error = h - y;
grad = (1/m) * error' * X;




% =============================================================

end
