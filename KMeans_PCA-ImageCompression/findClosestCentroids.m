function idx = findClosestCentroids(X, centroids)
%FINDCLOSESTCENTROIDS computes the centroid memberships for every example
%   idx = FINDCLOSESTCENTROIDS (X, centroids) returns the closest centroids
%   in idx for a dataset X where each row is a single example. idx = m x 1 
%   vector of centroid assignments (i.e. each entry in range [1..K])
%

% Set K
K = size(centroids, 1);

% You need to return the following variables correctly.
idx = zeros(size(X,1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Go over every example, find its closest centroid, and store
%               the index inside idx at the appropriate location.
%               Concretely, idx(i) should contain the index of the centroid
%               closest to example i. Hence, it should be a value in the 
%               range 1..K
%
% Note: You can use a for-loop over the examples to compute this.
%
m = size(X,1);
N = size(X,2);
distances = [];

% Finds the distance between each example and the different centroids. Assigns the
%example to the closest centroid
for i = 1:m
  for k = 1:K
    for n = 1:N
      dist(n) = (X(i,n) - centroids(k,n)).^2;
    end
    distances(k) = sum(dist);
  end
  [r,c] = min(distances);
  idx(i) = c;        
end
% =============================================================

end

