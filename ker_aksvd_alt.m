function [A, Z, errs, train_time] = ker_aksvd_alt( ...
    Y, A, D, n_nonzero_coefs_A, n_nonzero_coefs_D, max_iter_A, max_iter_D, ...
    sigma, ompparams, alpha, lambda, train_D, use_lin ...
)
%KER_AKSVD Summary of this function goes here
%   Detailed explanation goes here

    % Init params and vars
    n_samples = size(Y, 2);
    n_components_A = size(A, 2);
    n_components_D = size(D, 2);
    errs = zeros(1, max_iter_A);

    % Compute kernel
    tmp_time = tic;
    K_DD = zeros(n_components_D, n_components_D);
    for i = 1:n_components_D
       K_DD(i, :) = exp(-vecnorm(D(:, i) - D).^2 / sigma);
    end

    K_YD = zeros(n_samples, n_components_D);
    for i = 1:n_samples
        K_YD(i, :) = exp(-vecnorm(Y(:, i) - D).^2 / sigma);
    end
    kernel_time = toc(tmp_time);

    % start waitbar
    train_time = kernel_time;
    wb = waitbar(0, 'Training Kernel AKSVD-D...');

    for i_iter = 1:max_iter_A
        tmp_time = tic;

        % X coding method
        Z = omp_sparse(A'*K_YD', A'*K_DD*A, n_nonzero_coefs_A, ompparams{:});

        % optimize dictionary D
        P = A*Z;
        for j = 1:n_components_A     
            [~, data_indices, ~] = find(Z(j,:));
    
            if (isempty(data_indices))
                a = randn(n_components_D, 1);
                A(:,j) = a / sqrt(a'*K_DD*a);
            else
                R = P - A(:, j) * Z(j,:);
                a = (K_DD\K_YD' - R) * Z(j,:)' / (Z(j,:)*Z(j,:)');
                a = a / sqrt(a'*K_DD*a);
                z = (K_YD - R' * K_DD) * a;
                A(:, j) = a;
                Z(j, :) = z;
                P = R + A(:, j) * Z(j, :);
            end
        end

        if train_D
            % X coding method
            X = omp(Y, D, n_nonzero_coefs_D);

            % optimize dictionary D
            D = ker_update_D(Y, X, D, A, Z, sigma, alpha, lambda, max_iter_D, use_lin);
        end
        train_time = train_time + toc(tmp_time);

        % Compute current error
        if train_D
            % update kernels
            K_DD = zeros(n_components_D, n_components_D);
            for i = 1:n_components_D
               K_DD(i, :) = exp(-vecnorm(D(:, i) - D).^2 / sigma);
            end
        
            K_YD = zeros(n_samples, n_components_D);
            for i = 1:n_samples
                K_YD(i, :) = exp(-vecnorm(Y(:, i) - D).^2 / sigma);
            end
        end

        errs(i_iter) = trace(Z'*A'*K_DD*A*Z - 2*K_YD*A*Z);

        % update waitbar
        waitbar(i_iter/max_iter_A, wb, sprintf('Training Kernel AKSVD-D - Remaining time: %d [sec]',...
                round(train_time/i_iter*(max_iter_A - i_iter))));
    end
    
    % close waitbar
    close(wb);
end