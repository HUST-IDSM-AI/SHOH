% close all; clear; clc;
function SHOH(param, train, query)
close all

nchunks = floor(train.size / param.chunk_size);

%% parameters
param.alpha1 = 0.2;
param.alpha2 = 1 - param.alpha1;
param.eta = 10;
param.gamma = 1;
param.xi = 1;
param.mu = 1000;
param.max_iter = 7;

%% train
eva_info = cell(nchunks, 1);
eva_info_WRS = cell(nchunks, 1);

first_round = true;
for i = 1:nchunks
    idx_strt = (i - 1) * param.chunk_size + 1;
    if(i ~= nchunks)
        idx_end = idx_strt - 1 + param.chunk_size;
    else
        idx_end = train.size;
    end
    train = train_SHOH(param, train, idx_strt:idx_end, first_round);
    first_round = false;

    fprintf('-------------- Round / Total: %d / %d --------------\n', i, nchunks);
    eva_info{i, 1} = evaluate_perf(train.B', query.X * train.Wx', query.Y * train.Wy', train.L2(1:train.trained, :), query.L2);
    fprintf('MAP of SHOH in I->T: %.4g\n', eva_info{i, 1}.map_image2text);
    fprintf('MAP of SHOH in T->I: %.4g\n', eva_info{i, 1}.map_text2image);

    eva_info_WRS{i, 1} = evaluate_perf_WRS(train.B', query.X * train.Wx', query.Y * train.Wy', train.L2(1:train.trained, :), query.L2);
    fprintf('MAP of SHOH-WRS in I->T: %.4g\n', eva_info_WRS{i, 1}.map_image2text);
    fprintf('MAP of SHOH-WRS in T->I: %.4g\n', eva_info_WRS{i, 1}.map_text2image);
end
fprintf('----------------------- Done -----------------------\n');

%% save records
record_dir = fullfile(param.rec_dir, 'SHOH', param.ds_name);
if(~exist(record_dir, 'dir'))
    mkdir(record_dir);
end
record_name = ['test', num2str(param.t), ...
    '_QuerySize=', num2str(query.size), ...
    '_TrainSize=', num2str(train.size), ...
    '_ChunkSize=', num2str(param.chunk_size), ...
    '_NumBits=', num2str(param.nbits), ...
    '.mat'];
time = train.time;
save(fullfile(record_dir, record_name), 'param', 'eva_info', 'eva_info_WRS', 'time', '-v7.3');
end
