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

h = X * theta; % [examples' amount , 1]
delta = h - y; % [examples' amount , 1]
quadratic_delta = delta .^ 2; % [examples' amount , 1]
sum_delta = sum(quadratic_delta); % real number
result = sum_delta / (2 * m); % real number
quadratic_theta = theta .^ 2; % [features' amount + 1, 1]
temp_quad_theta = quadratic_theta(2: end); % [features' amount, 1]
reg_term = lambda * sum(temp_quad_theta) / (2 * m);
J = result + reg_term;

delta_time_X = delta' * X; % [1 , features' amount + 1]
lambda_theta = (lambda / m) * theta;
lambda_theta(1) = 0;
grad = (delta_time_X' / m) + lambda_theta; % [features' amount + 1 , 1]

% =========================================================================

grad = grad(:);

end
