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

H = sigmoid(X * theta);

costSum = sum(-y .* log(H) - (1 - y) .* log(1 - H));

theta_2 = theta .* theta;

theta_2(1) = 0;

theta_2_sum = sum(theta_2);

J = (1 / m) * costSum + (lambda / (2 * m)) * theta_2_sum;

gradSum = (H - y)' * X;

grad = ((1 / m) * gradSum)' + (lambda / m) * theta;
grad(1) = grad(1) - (lambda / m) * theta(1);

% =============================================================

end
