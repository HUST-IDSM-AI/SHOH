function [param, train, query] = split_dataset(X, Y, L, param)
% X: original features
% Y: original text
% L: original labels
% param.nquery: the number of test points
num_class1 = param.num_class1;
[N, ~] = size(L);

%% split test and training set
rng shuffle
param.seed = rng;
R = randperm(N);
nquery = param.nquery; iquery = R(1:nquery);
ntrain = N - nquery; itrain = R(nquery+1:N);

% randomize again
itrain = itrain(randperm(ntrain));
iquery = iquery(randperm(nquery));

% select training data
query.X = X(iquery, :);
query.Y = Y(iquery, :);
query.L1 = L(iquery, 1:num_class1);
query.L2 = L(iquery, num_class1+1:end);
query.size = nquery;

train.X = X(itrain, :);
train.Y = Y(itrain, :);
train.L1 = L(itrain, 1:num_class1);
train.L2 = L(itrain, num_class1+1:end);
train.size = ntrain;

end