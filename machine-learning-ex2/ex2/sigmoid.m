function g = sigmoid(z)
%SIGMOID Compute sigmoid function
%   g = SIGMOID(z) computes the sigmoid of z.

% You need to return the following variables correctly 
g = zeros(size(z));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the sigmoid of each value of z (z can be a matrix,
%               vector or scalar).

expNegZ = exp(-z);

denominator = 1 + expNegZ;

numerator = ones(size(z));

g = numerator ./ denominator;


% =============================================================

end
