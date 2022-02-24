function [k] = kernel_function(x, y, sigma)
    k = exp(-norm(x-y)^2 / sigma);
%     k =  (x'*y+sigma.a)^sigma.b;
end

