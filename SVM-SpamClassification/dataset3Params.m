function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% Matrix to try different values of C and sigma
Cmat= [0.01,0.03,0.1,0.3,1,3,10,30];
sigmaMat=[0.01,0.03,0.1,0.3,1,3,10,30];

% Get all possible combinations of C and sigma
[C1, sigma1] = meshgrid(Cmat, sigmaMat)
c = cat(2, C1', sigma1')
d = reshape(c,[],2)

errorMat= [];
for i = 1:size(d,1)
  model = svmTrain(X, y, d(i,1), @(x1, x2) gaussianKernel(x1, x2, d(i,2)));
  predictions = svmPredict(model, Xval);
  error = mean(double(predictions ~=yval));
  errorMat(i) = [error];
end  

[errval, ind] = min(errorMat');

% Choose the value for C and sigma which gives the lowest error
C = d(ind,1);
sigma = d(ind,2);
 



      
% =========================================================================

end
