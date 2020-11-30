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
                 hidden_layer_size, (input_layer_size + 1)); % hidden units' number * input units' number + 1

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1)); % output units' number * hidden units' number + 1

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

%% part1 the cost fuction
Y = zeros(m, num_labels);
for indexOfExamples = 1 : m 
    indexOfFeatures = y(indexOfExamples);
    Y(indexOfExamples, indexOfFeatures) = 1;
end

one_X = [ones(m, 1) X]; % [examples'number, input units' number + 1]
Z_input_hidden = one_X * Theta1'; % [examples' number, input units' number + 1] * [input units' number + 1, hidden units' number]
h_input_hidden = sigmoid(Z_input_hidden) % [examples' number, hidden units' number]

one_h_input_hidden = [ones(m, 1) h_input_hidden]; % [examples' number, hidden units' number + 1]
Z_hidden_output = one_h_input_hidden * Theta2'; % [examples' number, hidden units' number + 1] * [hidden units' number + 1, output units' number]
h_hidden_output = sigmoid(Z_hidden_output); % [examples' number, output units' number]

log_h_hidden_output1 = log(h_hidden_output);
former_unit = -Y .* log_h_hidden_output1; % [examples' number, output units' number] .* [examples' number, output units' number]
log_h_hidden_output2 = log(1 - h_hidden_output);
latter_unit = (1 - Y) .* log_h_hidden_output2; % [examples' number, output units' number] .* [examples' number, output units' number]
unit = former_unit - latter_unit; % [examples' number, output units' number]
sumAll = sum(unit, 'all');
J_nonReg = sumAll / m;

stdTheta1 = Theta1(:, 2:input_layer_size + 1);
stdTheta2 = Theta2(:, 2:hidden_layer_size + 1);
square_theta1 = stdTheta1 .^ 2;
square_theta2 = stdTheta2 .^ 2;
regularition_term = lambda * (sum(square_theta1, 'all') + sum(square_theta2, 'all')) / (2 * m);

J = J_nonReg + regularition_term;

%% part2 backforward
% 1 [examples' number, output units' number]
delta_out_hidden = h_hidden_output - Y;
% 2 [examples' number, output units' number] * [output units' numberm, hidden units' number + 1]
one_delta_hidden_in = (delta_out_hidden * Theta2) .* [zeros(m, 1) sigmoidGradient(Z_input_hidden)];
delta_hidden_in = one_delta_hidden_in(:, 2:end); % [examples' number, hidden units' number]
% 3 
delta1 = one_X' * delta_hidden_in; % [input units' number + 1, examples' number] * [examples' number, hidden units' number]
delta2 = one_h_input_hidden' * delta_out_hidden; % [hidden units' number + 1, examples' number] * [examples' number, output units' number]
% 4
Theta1_grad = delta1' / m; % [hidden units' number, input units' number + 1]
Theta1Reg = [zeros(hidden_layer_size, 1) (lambda / m) * Theta1(:, 2:end)];
Theta1_grad = Theta1_grad + Theta1Reg;
Theta2_grad = delta2' / m; % [output units' number, hidden units' number + 1]
Theta2Reg = [zeros(num_labels, 1) (lambda / m) * Theta2(:, 2:end)];
Theta2_grad = Theta2_grad + Theta2Reg;
% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
