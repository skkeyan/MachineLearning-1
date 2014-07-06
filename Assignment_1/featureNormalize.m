function [X_norm, mu, sigma] = featureNormalize(X)
%FEATURENORMALIZE Normalizes the features in X 
%   FEATURENORMALIZE(X) returns a normalized version of X where
%   the mean value of each feature is 0 and the standard deviation
%   is 1. This is often a good preprocessing step to do when
%   working with learning algorithms.

% You need to set these values correctly
%X_norm = X;
%mu = zeros(1, size(X, 2));
%sigma = zeros(1, size(X, 2));

% ====================== YOUR CODE HERE ======================
% Instructions: First, for each feature dimension, compute the mean
%               of the feature and subtract it from the dataset,
%               storing the mean value in mu. Next, compute the 
%               standard deviation of each feature and divide
%               each feature by it's standard deviation, storing
%               the standard deviation in sigma. 
%
%               Note that X is a matrix where each column is a 
%               feature and each row is an example. You need 
%               to perform the normalization separately for 
%               each feature. 
%
% Hint: You might find the 'mean' and 'std' functions useful.
%       

num_samples = rows(X);
num_features = columns(X);

X_norm = zeros(num_samples,num_features);
mu = zeros(1, num_features);
sigma = zeros(1, num_features);

for iter = 1:num_features

    mean_feature = mean(X(:,iter));
    sd_feature = std(X(:,iter));
    feature_norm = (X(:,iter) - mean_feature)/(sd_feature);
    
    %fprintf('Rows: %.0f, Columns: %.0f\n',rows(feature_norm), columns(feature_norm));
    %fprintf('[%f]\n', feature_norm);
    
    X_norm(:,iter) = feature_norm;
    mu(1,iter) = mean_feature;
    sigma(1,iter) = sd_feature;
    
    
end

%fprintf('Rows: %.0f, Columns: %.0f\n',rows(X_norm), columns(X_norm));
%fprintf('[%f]\n', X_norm);

% ============================================================

end
