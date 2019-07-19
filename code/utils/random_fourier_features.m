function Z = random_fourier_features(X, D, sigma)
    % creates Gaussian random features
    % Inputs:
    % X the datapoints to use to generate those features (d x n)
    % D the number of features to make
    % sigma the gaussian kernel parameter K(x,x') = exp(-|x-x'|^2/(2*sigma^2))
    % Output:
    % Z is (D x n)
    d = size(X, 1);
    W = normrnd(0, 1/sigma, [D, d]);
    b = 2*pi*rand(D, 1);
    Z = sqrt(2/D)*cos(W*X + b);
end