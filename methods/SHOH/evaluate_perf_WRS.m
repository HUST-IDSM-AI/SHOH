function eva = evaluate_perf_WRS(B, FxQuery, FyQuery, trainL_GT, queryL_GT)
BxTrain = B;
ByTrain = BxTrain;

BxQuery = sign(FxQuery);
ByQuery = sign(FyQuery);

% I->T
t = abs(FxQuery);
t(t >= 1) = 1;
WD = (repmat(sum(t, 2), 1, size(BxTrain, 1)) - t .* BxQuery * ByTrain') / 2;
[~, idx_rank] = sort(WD, 2);
eva.map_image2text = mAP(idx_rank', trainL_GT, queryL_GT);
[eva.precision_image2text, eva.recall_image2text] = precision_recall(idx_rank', trainL_GT, queryL_GT);

% T->I
t = abs(FyQuery);
t(t >= 1) = 1;
WD = (repmat(sum(t, 2), 1, size(ByTrain, 1)) - t .* ByQuery * BxTrain') / 2;
[~, idx_rank] = sort(WD, 2);
eva.map_text2image = mAP(idx_rank', trainL_GT, queryL_GT);
[eva.precision_text2image, eva.recall_text2image] = precision_recall(idx_rank', trainL_GT, queryL_GT);

end

