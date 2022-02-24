clc;
clear;
close all;

DataPath = 'Digits';
% DataPath = 'MNIST';
% DataPath = 'CIFAR-10';

switch DataPath
    case 'Digits'
        % Load training and testing data
        [XTrain,YTrain,anglesTrain] = digitTrain4DArrayData;
        [XTest,YTest,anglesTest] = digitTest4DArrayData;

        % Prepare data for training
        Y = reshape(XTrain, 28*28, []);

        % Digits params
        alpha = 0.0005;
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
        alpha = 0.0005;
    case 'CIFAR-10'
        % Load training and testing data
        [XTrain,YTrain,XValidation,YValidation] = loadCIFARData_origin('dbs');
        
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
        alpha = 0.0006;
    otherwise
        fprintf('Error! Database not found!')
        return;
end

% normalize signals
Y = normc(Y);

% DL parameters
sigma = 10;
lambda = 1;
train_D = 1;
n_rounds = 10;
ompparams = {'checkdict', 'off'};

max_iter_D = 3;
n_components_D = 50;
n_nonzero_coefs_D = 5;

max_iter_A = 10;
n_components_A = 20;
n_nonzero_coefs_A = 4;
n_samples = size(Y, 2);

r_train_time0 = 0;
r_train_time1 = 0;
r_train_time2 = 0;
r_train_time3 = 0;

r_errs0 = zeros(1, max_iter_A);
r_errs1 = zeros(1, max_iter_A);
r_errs2 = zeros(1, max_iter_A);
r_errs3 = zeros(1, max_iter_A);

for n_round = 1:n_rounds
    % Prepare dictionary D
    tic
    if train_D
        D = aksvd(Y, 10, n_components_D, n_nonzero_coefs_D);
    % else
    %     rp = randperm(size(Y, 2));
    %     D = Y(:, rp(1:n_components_D));
    end
    train_time_D = toc;
    
    % Run standard Kernel AK-SVD
    disp('Standard Kernel AK-SVD')
    A = normcol_equal(randn(n_samples, n_components_A));
    
    [A0, X0, errs0, train_time0] = ker_aksvd(Y, A, n_nonzero_coefs_A, ...
                                             max_iter_A, sigma);
    
    % Run Kernel AK-SVD-D
    disp('Standard Kernel AK-SVD-D')
    A = normcol_equal(randn(n_components_D, n_components_A));
    
    [A1, Z1, errs1, train_time1] = ker_aksvd_alt(...
        Y, A, D, n_nonzero_coefs_A, n_nonzero_coefs_D, max_iter_A, max_iter_D, ...
        sigma, ompparams, alpha, lambda, 0, 0 ...
    );
    train_time1 = train_time1 + train_time_D;
    
    % Run Kernel AK-SVD-D with training D
    disp('Standard Kernel AK-SVD-D trained D')
    [A2, Z2, errs2, train_time2] = ker_aksvd_alt(...
        Y, A, D, n_nonzero_coefs_A, n_nonzero_coefs_D, max_iter_A, max_iter_D, ...
        sigma, ompparams, alpha, lambda, 1, 0 ...
    );
    train_time2 = train_time2 + train_time_D;
    
    % Run Kernel AK-SVD-D with training lin D
    disp('Standard Kernel AK-SVD-D trained D')
    [A3, Z3, errs3, train_time3] = ker_aksvd_alt(...
        Y, A, D, n_nonzero_coefs_A, n_nonzero_coefs_D, max_iter_A, max_iter_D, ...
        sigma, ompparams, alpha, lambda, 1, 0 ...
    );
    train_time3 = train_time3 + train_time_D;
    
    % K_YY trace
    s = 0;
    for i = 1:n_samples
        s = s + kernel_function(Y(:, i), Y(:, i), sigma);
    end
    
    errs0 = sqrt(errs0 + s); % / (n_samples);
    errs1 = sqrt(errs1 + s); % / (n_samples);
    errs2 = sqrt(errs2 + s); % / (n_samples);
    errs3 = sqrt(errs3 + s); % / (n_samples);

    r_train_time0 = r_train_time0 + train_time0;
    r_train_time1 = r_train_time1 + train_time1;
    r_train_time2 = r_train_time2 + train_time2;
    r_train_time3 = r_train_time3 + train_time3;

    r_errs0 = r_errs0 + errs0;
    r_errs1 = r_errs1 + errs1;
    r_errs2 = r_errs2 + errs2;
    r_errs3 = r_errs3 + errs3;
end

r_train_time0 = r_train_time0 / n_rounds;
r_train_time1 = r_train_time1 / n_rounds;
r_train_time2 = r_train_time2 / n_rounds;
r_train_time3 = r_train_time3 / n_rounds;

r_errs0 = r_errs0 / n_rounds;
r_errs1 = r_errs1 / n_rounds;
r_errs2 = r_errs2 / n_rounds;
r_errs3 = r_errs3 / n_rounds;

figure;
hold on;
plot(1:max_iter_A, r_errs0);
plot(1:max_iter_A, r_errs1);
plot(1:max_iter_A, r_errs2);
plot(1:max_iter_A, r_errs3);
xlabel('iter')
ylabel('err')
legend(sprintf('KDL err=%0.4f time=%0.4f', r_errs0(end), r_train_time0), ...
       sprintf('RKDL-D err=%0.4f time=%0.4f', r_errs1(end), r_train_time1), ...
       sprintf('RKDL-trD err=%0.4f time=%0.4f', r_errs2(end), r_train_time2), ...
       sprintf('RKDL-lintrD err=%0.4f time=%0.4f', r_errs3(end), r_train_time3))

save(strcat(DataPath, '_results'), 'r_errs0', 'r_errs1', 'r_errs2', 'r_errs3', ...
            'r_train_time0', 'r_train_time1', 'r_train_time2', 'r_train_time3')


function [XTrain,YTrain,XTest,YTest] = loadCIFARData_origin(location)

location = fullfile(location,'cifar-10-batches-mat');

[XTrain1,YTrain1] = loadBatchAsFourDimensionalArray(location,'data_batch_1.mat');
[XTrain2,YTrain2] = loadBatchAsFourDimensionalArray(location,'data_batch_2.mat');
[XTrain3,YTrain3] = loadBatchAsFourDimensionalArray(location,'data_batch_3.mat');
[XTrain4,YTrain4] = loadBatchAsFourDimensionalArray(location,'data_batch_4.mat');
[XTrain5,YTrain5] = loadBatchAsFourDimensionalArray(location,'data_batch_5.mat');
XTrain = cat(4,XTrain1,XTrain2,XTrain3,XTrain4,XTrain5);
YTrain = [YTrain1;YTrain2;YTrain3;YTrain4;YTrain5];

[XTest,YTest] = loadBatchAsFourDimensionalArray(location,'test_batch.mat');
end

function [XBatch,YBatch] = loadBatchAsFourDimensionalArray(location,batchFileName)
s = load(fullfile(location,batchFileName));
XBatch = s.data';
XBatch = reshape(XBatch,32,32,3,[]);
XBatch = permute(XBatch,[2 1 3 4]);
YBatch = convertLabelsToCategorical(location,s.labels);
end

function categoricalLabels = convertLabelsToCategorical(location,integerLabels)
s = load(fullfile(location,'batches.meta.mat'));
categoricalLabels = categorical(integerLabels,0:9,s.label_names);
end
