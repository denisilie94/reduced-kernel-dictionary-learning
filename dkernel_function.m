function [dK] = dkernel_function(d, Y, sigma, kernel_type)
    switch kernel_type
        case 'rbf'
            dK = (- exp(-vecnorm(d - Y).^2 / sigma) / sigma .* (d - Y))';
        case 'poly'
            dK = -((sigma.b * (d'*Y + sigma.a).^(sigma.b - 1)).*Y)';
    end
end
