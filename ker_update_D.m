function [D] = ker_update_D(Y, X, D, A, Z, sigma, alpha, lambda, max_iter, use_lin)
    % n_components -  the number of atoms
    % n_samples - number of atoms
    % sigma - rbf parameter

    n_features = size(Y, 1);
    n_components = size(D, 2);    

    C = A*Z;
    CCT = C*C';

    for iter = 1:max_iter
        for i = 1:n_components
            dK1 = (- 2 * exp(-vecnorm(D(:, i) - D).^2 / sigma) / sigma .* (D(:, i) - D))';
            dfK1 = repmat(CCT(i, :)', 1, n_features) .* dK1;
            dfK1 = dfK1 + repmat(CCT(:, i), 1, n_features) .* dK1;
            dfK1 = dfK1 - CCT(i, i) * dK1(i, :); 
            dfK1 = sum(dfK1)';

            % -----------------------------------

            dK2 = (- 2 * exp(-vecnorm(D(:, i) - Y).^2 / sigma) / sigma .* (D(:, i) - Y))';
            dfK2 = repmat(C(i, :)', 1, n_features) .* dK2;
            dfK2 = sum(dfK2)';

            % -----------------------------------
            
            if use_lin
                dE = - (Y - D * X)*X(i, :)';
            else
                dE = 0; 
            end

            D(:, i) = D(:, i) - alpha * ((dfK1 - 2*dfK2) - lambda*dE);
        end
    end
end