close all; clear; clc;
% parpool('local', 5);

addpath(genpath('./methods/'));
addpath(genpath('./utils/'));

param.ds_dir = './datasets/';
param.rec_dir = './results/';

ds_name = {'FashionVC', 'Ssense'};
nbits = [16 32 64 128];
test_times = 5;

% spmd
%     param.t = labindex;
for t = 1:test_times
    param.t = t;
    % DATASET
    for ds = 1:length(ds_name)
        param.ds_name = ds_name{ds};
        [param, train, query] = load_dataset(param);
        % CODE LENGTH
        for nb = 1:length(nbits)
            param.nbits = nbits(nb);
            fprintf('LOAD DATASET: %s\n', param.ds_name);
            fprintf('CODE LENGTH: %d\n',param.nbits);
            SHOH(param, train, query);
        end
    end
end