clc;
clear;
close all;

% DataPath = 'Digits';
% DataPath = 'MNIST';
DataPath = 'CIFAR-10';

% kernel_type = 'rbf';
kernel_type = 'poly';

switch DataPath
    case 'Digits'
        % Load training and testing data
        [XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
        [XTest,YTest,anglesTest] = digitTest4DArrayData;

        % Prepare data for training
        Y = reshape(XTrain, 28*28, []);

        % Digits params
        if strcmp(kernel_type, 'rbf')
            alpha = 0.0006;
        else
            alpha = 1e-7;
        end
    case 'MNIST'
        % Load training and testing data
        oldpath = addpath(fullfile(matlabroot,'examples','nnet','main'));
        filenameImagesTrain = 'dbs/train-images-idx3-ubyte.gz';
        filenameLabelsTrain = 'dbs/train-labels-idx1-ubyte.gz';
        
        XTrain = extractdata(processImagesMNIST(filenameImagesTrain));
        YTrain = grp2idx(processLabelsMNIST(filenameLabelsTrain));

        % Prepare data for training
        label = 5;
        Y = XTrain(:, :, YTrain == label);
        Y = reshape(Y, 28*28, []);
        
        % MNIST params
        if strcmp(kernel_type, 'rbf')
            alpha = 0.0005;
        else
            alpha = 1e-7;
        end
    case 'CIFAR-10'
        % Load training and testing data
        [XTrain,YTrain,XValidation,YValidation] = loadCIFARData('dbs');
        
        % Convert categorical to double
        YTrain = grp2idx(YTrain);

        % Extract all images as doubles
        Y = zeros(32, 32, size(XTrain, 4));
        for j = 1:size(XTrain, 4)
            Y(:, :, j) = im2double(rgb2gray(XTrain(:, :, :, j)));
        end

        % Prepare data for training
        label = 1;
        Y = Y(:, :, YTrain == label);
        Y = reshape(Y, 32*32, []);

        % CIFAR-10 params
        if strcmp(kernel_type, 'rbf')
            alpha = 0.0006;
        else
            alpha = 1e-7;
        end
    otherwise
        fprintf('Error! Database not found!')
        return;
end

% normalize signals
Y = normc(Y);

% rbf params
% sigma = 1;

% poly params
sigma = {};
sigma.a = 0;
sigma.b = 2;

max_iter = 10;
n_components = 128;
n_features = size(Y, 1);
n_samples = size(Y, 2);

D = randn(n_features, n_components); D = normc(D);
C = randn(n_components, n_samples);

K_DD = kernel_function(D, D, sigma, kernel_type);
K_YD = kernel_function(Y, D, sigma, kernel_type);

% C = A*X;
CCT = C*C';

error = zeros(1, max_iter + 1);
error(1) = trace(C'*K_DD*C - 2*K_YD*C);

for iter = 1:max_iter
    disp(iter);
    for i = 1:n_components       
        dK1 = dkernel_function(D(:, i), D, sigma, kernel_type);
        dfK1 = repmat(CCT(i, :)', 1, n_features) .* dK1;
        dfK1 = dfK1 + repmat(CCT(:, i), 1, n_features) .* dK1;
        dfK1 = dfK1 - CCT(i, i) * dK1(i, :); 
        dfK1 = sum(dfK1)';

        % -----------------------------------

        dK2 = dkernel_function(D(:, i), Y, sigma, kernel_type);
        dfK2 = repmat(C(i, :)', 1, n_features) .* dK2;
        dfK2 = sum(dfK2)';
       
        D(:, i) = D(:, i) - alpha * (dfK1 - 2*dfK2);
    end

    K_DD = kernel_function(D, D, sigma, kernel_type);
    K_YD = kernel_function(Y, D, sigma, kernel_type);
    error(iter + 1) = trace(C'*K_DD*C - 2*K_YD*C);
end

figure
plot(error)
title('Error evolution')
