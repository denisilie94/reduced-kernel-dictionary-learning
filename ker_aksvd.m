function [A, X, errs, train_time] = ker_aksvd(Y, A, n_nonzero_coefs, max_iter, sigma)
%KER_AKSVD Summary of this function goes here
%   Detailed explanation goes here

    % Init params and vars
    n_samples = size(Y, 2);
    errs = zeros(1, max_iter);

    % Compute kernel
    tmp_time = tic;
    K = zeros(n_samples, n_samples);
    for i = 1:n_samples
       K(i, :) = exp(-vecnorm(Y(:, i) - Y).^2 / sigma);
    end
    kernel_time = toc(tmp_time);

    % start waitbar
    train_time = kernel_time;
    wb = waitbar(0, 'Training Kernel AKSVD...');

    for i_iter = 1:max_iter
        tmp_time = tic;

        % X coding method
        X = omp_ker(K, [], A, n_nonzero_coefs, []);

        % optimize dictionary D
        E = eye(n_samples) - A*X;
        for j = 1:size(A,2)     
            [~, data_indices, x] = find(X(j,:));

            if (isempty(data_indices))
                a = randn(n_samples,1);
                A(:,j) = a / sqrt(a'*K*a);
            else
                F = E(:, data_indices) + A(:, j) * x;
                a = F*x';
                A(:, j) = a / sqrt(a'*K*a);
                X(j, data_indices) = F'*K*A(:, j);
                E(:, data_indices) = F - A(:, j)*X(j, data_indices);
            end
        end
        train_time = train_time + toc(tmp_time);

        % Compute current error
        errs(i_iter) = trace(X'*A'*K*A*X - 2*K*A*X);

        % update waitbar
        waitbar(i_iter/max_iter, wb, sprintf('Training Kernel AKSVD - Remaining time: %d [sec]',...
                round(train_time/i_iter*(max_iter - i_iter))));
    end
    
    % close waitbar
    close(wb);
end

