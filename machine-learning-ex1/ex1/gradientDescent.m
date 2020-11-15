function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

    H = X * theta;
    different = H - y;
    % theta = 0
    unit0 = different .* X(:, 1);
    sum0 = sum(unit0);
    part0 = alpha * sum0 / m;
    % theta = 1
    unit1 = different .* X(:, 2);
    sum1 = sum(unit1);
    part1 = alpha * sum1 / m;
    
    part = [part0; part1];
    theta2 = theta - part;
    theta = theta2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
