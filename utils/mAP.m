function map = mAP(ids, Lbase, Lquery)

nquery = size(ids, 2);
APx = zeros(nquery, 1);

% 检索集的大小
R = size(Lbase,1); % Configurable

for i = 1 : nquery
    % 第n个查询点的标记
    label = Lquery(i, :);
    label(label == 0) = -1;

    % 取第n个查询点的邻居关系
    idx = ids(:, i);

    % 判断邻居关系中，与查询点是否属于同一类别，并保持亲疏关系
    imatch = sum(bsxfun(@eq, Lbase(idx(1:R), :), label), 2) > 0;

    % 统计正样本个数
    LX = sum(imatch);

    % 累加TP
    Lx = cumsum(imatch);
    Px = Lx ./ (1:R)';
    if LX ~= 0
        APx(i) = sum(Px .* imatch) / LX;
    end
end
map = mean(APx);

end
