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

distance_matrix = zeros(size(X,1), K);
for centroidIdx = 1:K,
  % Calculate the distance to the centroid k in a vectorised way
  % Compute the sum of squared differences between each row in X and the 
  % centroid k. The second parameter in the sum function set to '2' indicates
  % that the sum has to be done for each row.
  v_distance = sum(bsxfun(@minus, X, centroids(centroidIdx,:)).^2,2);
  distance_matrix(:, centroidIdx) = v_distance;  
end
% Get the index of the minimum values for each of the centroids distances.
% I had to transpose the distance matrix as min calculates the minimum per 
% column and not per row.
[~, min_index] = min(distance_matrix');
idx = min_index(:);

% =============================================================

end

