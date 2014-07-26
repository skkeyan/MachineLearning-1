function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%

% Add ones to the X data matrix
X = [ones(m, 1) X];

% Computing (a) for Layer 2

z_matrix_layer_2 = Theta1 * X';
a_matrix_layer_2 = sigmoid(z_matrix_layer_2);

a_matrix_layer_2_transpose = a_matrix_layer_2';
num_rows_temp = size(a_matrix_layer_2_transpose,1);
a_matrix_layer_2_transpose = [ones(num_rows_temp,1),a_matrix_layer_2_transpose];
a_matrix_layer_2 = a_matrix_layer_2_transpose';  % Size is 26 * 5000

% Computing (a) for Layer 3 - This is the output layer (prediction layer) of the neural network

z_matrix_layer_3 = Theta2 * a_matrix_layer_2;
a_matrix_layer_3 = sigmoid(z_matrix_layer_3);

% Create the prediction matrix

prediction_matrix = a_matrix_layer_3';
[val,index] = max(prediction_matrix,[],2);
p = index;

% =========================================================================


end
