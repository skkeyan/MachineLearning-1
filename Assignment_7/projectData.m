function Z = projectData(X, U, K)
%PROJECTDATA Computes the reduced data representation when projecting only 
%on to the top k eigenvectors
%   Z = projectData(X, U, K) computes the projection of 
%   the normalized inputs X into the reduced dimensional space spanned by
%   the first K columns of U. It returns the projected examples in Z.
%

% You need to return the following variables correctly.
Z = zeros(size(X, 1), K);

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the projection of the data using only the top K 
%               eigenvectors in U (first K columns). 
%               For the i-th example X(i,:), the projection on to the k-th 
%               eigenvector is given as follows:
%                    x = X(i, :)';
%                    projection_k = x' * U(:, k);
%

m = size(X,1);   % Number of training examples
U_reduce = U(:,1:K);  % U_reduce is a matrix of size n * K

for i = 1:m  % Iterate over m examples
    x_ith_row = X(i,:)';  % Transpose of each row in X matrix gives a vector that is of size n * 1 that represents a row in X
    projection_k = U_reduce' * x_ith_row;  % projection_k is a K*1 vector as U_reduce is matrix of size (n * K) and x_ith_row is (n * 1) vector
    Z(i,:) = projection_k';  % Each row of X is transformed into a K-dimensional vector
end



% =============================================================

end
