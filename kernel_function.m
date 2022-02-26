function [K] = kernel_function(X, Y, sigma, kernel_type)
    K = zeros(size(X, 2), size(Y, 2));
    switch kernel_type
        case 'rbf'
            for i = 1:size(X, 2)
               K(i, :) = exp(-vecnorm(X(:, i) - Y).^2 / sigma);
            end
        case 'poly'
            K = (X'*Y + sigma.a).^sigma.b;
    end
end