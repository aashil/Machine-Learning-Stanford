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
temp1 = 0;
temp2 = 0;
temp3 = 0;
for i=1:m
	
	J = J + (-y(i)*log(sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3))) - (1 - y(i))*log(1 - sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3))));
	temp1 = temp1 + (sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3)) - y(i))*X(i,1);
	temp2 = temp2 + (sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3)) - y(i))*X(i,2);
	temp3 = temp3 + (sigmoid(theta(1)*X(i,1) + theta(2)*X(i,2) + theta(3)*X(i,3)) - y(i))*X(i,3);
% =============================================================

end
temp4 = 0;
for j=2:size(theta);
	
	temp4 = temp4 + theta(j)^2;
	
end
grad(1) = temp1/m;
grad(2) = temp2/m + (lambda/m)*theta(2); 
grad(3) = temp3/m + (lambda/m)*theta(3);
J = J / m + (lambda/(2*m)) * temp4;





% =============================================================

end
