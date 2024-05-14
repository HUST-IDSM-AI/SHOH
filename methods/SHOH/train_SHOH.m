function train = train_SHOH(param, train, idx, first_round)
tic;
r = param.nbits;
num_class1 = param.num_class1;
num_class2 = param.num_class2;
dx = size(train.X, 2);
dy = size(train.Y, 2);
n_t = numel(idx);

%% hyperparameters
eta = param.eta;
alpha1 = param.alpha1;
alpha2 = param.alpha2;
gamma = param.gamma;
xi = param.xi;
mu = param.mu;
max_iter = param.max_iter;

X_new = train.X(idx, :)';          % image feature vector
Y_new = train.Y(idx, :)';          % text feature vector
L1_new = train.L1(idx, :);         % labels at the 1st layer
L2_new = train.L2(idx, :);         % labels at the 2nd layer
A1_2 = train.A1_2;                 % cross-layer affiliation

if first_round == true
    train.B = [];
    train.trained = 0;
    train.time.train_time = [];
    train.cnt1 = zeros(num_class1, 1);
    train.cnt2 = zeros(num_class2, 1);

    %% initialize tempory variables
    T = cell(1, 11);
    % D2 = B_old * S2_old + B_new * S2_new
    T{1, 1} = zeros(r, num_class2);
    % D1 = B_old * S1_old + B_new * S1_new
    T{1, 2} = zeros(r, num_class1);
    % E = b_old * B_old_' + b_new * B_new_'
    T{1, 3} = zeros(r, r - 1);
    % F1 = B_old * X_old' + B_new * X_new'
    T{1, 4} = zeros(r, dx);
    % F2 = B_old * Y_old' + B_new * Y_new'
    T{1, 5} = zeros(r, dy);
    % G1 = X_old * X_old' + X_new * X_new'
    T{1, 6} = zeros(dx, dx);
    % G2 = Y_old * Y_old' + Y_new * Y_new'
    T{1, 7} = zeros(dy, dy);
    % meanX1
    T{1, 8} = zeros(dx, num_class1);
    % meanX2
    T{1, 9} = zeros(dx, num_class2);
    % meanY1
    T{1, 10} = zeros(dy, num_class1);
    % meanY2
    T{1, 11} = zeros(dy, num_class2);

    %% initialize C1,  C2 randomly
    C1 = sign(randn(r, num_class1)); C1(C1 == 0) = -1;
    C2 = sign(randn(r, num_class2)); C2(C2 == 0) = -1;

else
    T = train.T;
    C1 = train.C1;
    C2 = train.C2;
end

%% initialize B_new randomly
B_new = randn(r, n_t); B_new(B_new == 0) = -1;

%% soft similarity labels constructing
S1_new = L1_new;
V2_new = L1_new * A1_2 + train.L2(idx, :);
S2_new = zeros(size(L2_new));
for row = 1:n_t
    S2_new(row, :) = V2_new(row, :) / norm(V2_new(row, :)) + gamma * L2_new(row, :);
end

%% online hash code learning
for i = 1:max_iter
    %% B_new-step
    P = r * alpha1 * C1 * S1_new' + r * alpha2 * C2 * S2_new';
    for l = 1:r
        idx_exc = setdiff(1:r, l);
        p = P(l, :);
        c1 = C1(l, :); C1_ = C1(idx_exc, :);
        c2 = C2(l, :); C2_ = C2(idx_exc, :);
        B_new_ = B_new(idx_exc, :);

        b_new = sign(p - (alpha1 * c1 * C1_' + alpha2 * c2 * C2_') * B_new_);b_new(b_new == 0) = -1;
        B_new(l, :) = b_new;
    end

    %% C2-step
    Q2 = r * (alpha2 * (B_new * S2_new + T{1, 1}) + eta * C1 * A1_2);
    for l = 1:r
        idx_exc = setdiff(1:r, l);
        q2 = Q2(l, :);
        b_new = B_new(l, :); B_new_ = B_new(idx_exc, :);
        c1 = C1(l, :); C1_ = C1(idx_exc, :);
        C2_ = C2(idx_exc, :);

        c2 = sign(q2 - (alpha2 * b_new * B_new_' + alpha2 * T{1, 3}(l, :) + eta * c1 * C1_') * C2_); c2(c2 == 0) = -1;
        C2(l, :) = c2;
    end

    %% C1-step
    Q1 = r * (alpha1 * (B_new * S1_new + T{1, 2}) + eta * C2 * A1_2');
    for l = 1:r
        idx_exc = setdiff(1:r, l);
        b_new = B_new(l, :); B_new_ = B_new(idx_exc, :);
        C1_ = C1(idx_exc, :);
        c2 = C2(l, :); C2_ = C2(idx_exc, :);
        q1 = Q1(l, :);

        c1 = sign(q1 - (alpha1 * b_new * B_new_' + alpha1 * T{1, 3}(l, :) + eta * c2 * C2_') * C1_);c1(c1 == 0) = -1;
        C1(l, :) = c1;
    end
end

train.B = [train.B B_new];
T{1, 1} = T{1, 1} + B_new * S2_new;
T{1, 2} = T{1, 2} + B_new * S1_new;

for l = 1:r
    idx_exc = setdiff(1:r, l);
    b = B_new(l, :);
    B_ = B_new(idx_exc, :);
    T{1, 3}(l, :) = T{1, 3}(l, :) + b * B_';
end
T{1, 4} = double(T{1, 4} + B_new * X_new');
T{1, 5} = double(T{1, 5} + B_new * Y_new');
T{1, 6} = double(T{1, 6} + X_new * X_new');
T{1, 7} = double(T{1, 7} + Y_new * Y_new');

%% online hash function learning
for i = 1:num_class1
    ddx = find(L1_new(:, i) == 1);
    cnt = numel(ddx);
    if cnt > 0
        T{1, 8}(:, i) = double((T{1, 8}(:, i) * train.cnt1(i) + sum(X_new(:, ddx), 2)) ./ (train.cnt1(i) + numel(ddx)));
        T{1, 10}(:, i) = double((T{1, 10}(:, i) * train.cnt1(i) + sum(Y_new(:, ddx), 2)) ./ (train.cnt1(i) + numel(ddx)));
        train.cnt1(i) = train.cnt1(i) + numel(ddx);
    end
end

for i = 1:num_class2
    ddx = find(L2_new(:, i) == 1);
    cnt = numel(ddx);
    if cnt > 0
        T{1, 9}(:, i) = double((T{1, 9}(:, i) * train.cnt2(i) + sum(X_new(:, ddx), 2)) ./ (train.cnt2(i) + numel(ddx)));
        T{1, 11}(:, i) = double((T{1, 11}(:, i) * train.cnt2(i) + sum(Y_new(:, ddx), 2)) ./ (train.cnt2(i) + numel(ddx)));
        train.cnt2(i) = train.cnt2(i) + numel(ddx);
    end
end

Wx = (T{1, 4} + mu * alpha1 * C1 * T{1, 8}' + mu * alpha2 * C2 * T{1, 9}') / (T{1, 6} + mu * alpha1 * T{1, 8} * T{1, 8}' + mu * alpha2 * T{1, 9} * T{1, 9}' + xi * eye(dx, dx));
Wy = (T{1, 5} + mu * alpha1 * C1 * T{1, 10}' + mu * alpha2 * C2 * T{1, 11}') / (T{1, 7} + mu * alpha1 * T{1, 10} * T{1, 10}' + mu * alpha2 * T{1, 11} * T{1, 11}' + xi * eye(dy, dy));

train.time.train_time = [train.time.train_time; toc];
train.trained = train.trained + n_t;

%% return variables
train.T = T;
train.Wx = Wx;
train.Wy = Wy;
train.C1 = C1;
train.C2 = C2;

end