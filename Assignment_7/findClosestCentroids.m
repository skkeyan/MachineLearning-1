function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%

num_training_examples = size (X,1);
num_features = size(X,2);

%fprintf(' Number of training examples:%d', num_training_examples);
%fprintf(' Number of features:%d', num_features);
%fprintf(' Number of centroids:%d', K);

for i = 1:num_training_examples

    distance = 9999999; % Arbitrary high number
    idx(i) = 1;
    
    for j = 1:K
        difference = X(i,:) - centroids(j,:);
        distance_temp = difference * difference';
       
        if(distance_temp < distance)
            distance = distance_temp;
            idx(i) = j;
        endif
    end
end



% =============================================================

end

