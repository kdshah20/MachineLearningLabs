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

% predicted value of htheta
h = sigmoid(X*theta);

%parameter vector theta excluding theta0
theta_others = theta(2:size(theta,1));

%cost without regularization
J_initial= (1/m) *((-y)'*log(h) - (ones(m,1) -y)'*log(ones(size(h))-h));

% regularization parameter for cost and gradient
reg_J =  (lambda/(2*m))* theta_others.^2;
reg_grad = lambda/m * theta_others;

% final cost and gradient with regularization
J = J_initial + sum(reg_J);
grad = ((1/m)*X' * (h - y) ) + [0;reg_grad];

%
% =============================================================
end
