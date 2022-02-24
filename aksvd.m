function [D] = aksvd(Y, max_iter, n_components, n_nonzero_coefs)
%AKSVD Summary of this function goes here
%   Detailed explanation goes here

    % start waitbar
    train_time = 0;
    wb = waitbar(0, 'Training AKSVD...');

    n_features = size(Y, 1);
    D = normcol_equal(randn(n_features, n_components));
    
    for i_iter = 1:max_iter
        tmp_time = tic;

        % X coding method
        X = omp(Y, D, n_nonzero_coefs);

        % optimize dictionary D
        E = Y - D*X;
        for j = 1:size(D,2)     
            [~, data_indices, x] = find(X(j,:));

            if (isempty(data_indices))
                d = randn(size(D,1), 1);
                D(:, j) = d / norm(d);
            else
                F = E(:, data_indices) + D(:, j) * x;
                d = F*x';
                D(:, j) = d / norm(d);
                X(j, data_indices) = F'*D(:, j);
                E(:, data_indices) = F - D(:, j)*X(j, data_indices);
            end
        end

        train_time = train_time + toc(tmp_time);

        % update waitbar
        waitbar(i_iter/max_iter, wb, sprintf('Training AKSVD - Remaining time: %d [sec]',...
                round(train_time/i_iter*(max_iter - i_iter))));
    end

    % close waitbar
    close(wb);
end
