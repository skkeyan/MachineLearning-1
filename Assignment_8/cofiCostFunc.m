function [J, grad] = cofiCostFunc(params, Y, R, num_users, num_movies, ...
                                  num_features, lambda)
%COFICOSTFUNC Collaborative filtering cost function
%   [J, grad] = COFICOSTFUNC(params, Y, R, num_users, num_movies, ...
%   num_features, lambda) returns the cost and gradient for the
%   collaborative filtering problem.
%

% Unfold the U and W matrices from params
X = reshape(params(1:num_movies*num_features), num_movies, num_features);
Theta = reshape(params(num_movies*num_features+1:end), ...
                num_users, num_features);

            
% You need to return the following values correctly
J = 0;
X_grad = zeros(size(X));
Theta_grad = zeros(size(Theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost function and gradient for collaborative
%               filtering. Concretely, you should first implement the cost
%               function (without regularization) and make sure it is
%               matches our costs. After that, you should implement the 
%               gradient and use the checkCostFunction routine to check
%               that the gradient is correct. Finally, you should implement
%               regularization.
%
% Notes: X - num_movies  x num_features matrix of movie features
%        Theta - num_users  x num_features matrix of user features
%        Y - num_movies x num_users matrix of user ratings of movies
%        R - num_movies x num_users matrix, where R(i, j) = 1 if the 
%            i-th movie was rated by the j-th user
%
% You should set the following variables correctly:
%
%        X_grad - num_movies x num_features matrix, containing the 
%                 partial derivatives w.r.t. to each element of X
%        Theta_grad - num_users x num_features matrix, containing the 
%                     partial derivatives w.r.t. to each element of Theta
%

pred_rating = X * Theta';  % Gives a 1682 x 943 size matrix

% Setting the values in predicted rating and actual rating to zero where R(i,j) = 0, i.e. no rating has been provided by the user
pred_rating_by_R = R .* pred_rating;
actual_rating_by_R = R .* Y;
error_in_rating = pred_rating_by_R - actual_rating_by_R;

% Computing the cost function by squaring the error terms between predicted and actual rating
J_first_term = (1/2) * sum(sum((error_in_rating).^2)); % Cost Function without regularization
J_second_term = (lambda/2) * sum(sum(Theta.^2));       % Regularization term for Theta
J_third_term = (lambda/2) * sum(sum(X.^2));            % Regularization term for X 

J = J_first_term + J_second_term + J_third_term;       % Total cost function

% Computing the Gradient for Theta

Theta_grad_first_term = error_in_rating' * X;   % Gradient for Theta is matrix of size 943 x 100. size(error_in_rating) = [1682,943] and size(X) = [1682,100]
Theta_grad_regularization_term = lambda * Theta;
Theta_grad = Theta_grad_first_term + Theta_grad_regularization_term;


% Computing the Gradient for X
X_grad_first_term = error_in_rating * Theta;    % Gradient for X is matrix of size 1682 x 100. size(error_in_rating) = [1682,943] and size(Theta) = [943,100]
X_grad_regularization_term = lambda * X;
X_grad = X_grad_first_term + X_grad_regularization_term;

% =============================================================

grad = [X_grad(:); Theta_grad(:)];

end
