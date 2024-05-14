function eva = evaluate_perf(B, FxQuery, FyQuery, trainL_GT, queryL_GT)
BxTrain = compactbit(B > 0);
ByTrain = BxTrain;

BxQuery = compactbit(sign(FxQuery) > 0);
ByQuery = compactbit(sign(FyQuery) > 0);

% I->T
D = hammingDist(BxQuery, ByTrain);
[~, idx_rank] = sort(D, 2);
eva.map_image2text = mAP(idx_rank', trainL_GT, queryL_GT);
[eva.precision_image2text, eva.recall_image2text] = precision_recall(idx_rank', trainL_GT, queryL_GT);

% T->I
D = hammingDist(ByQuery, BxTrain);
[~, idx_rank] = sort(D, 2);
eva.map_text2image = mAP(idx_rank', trainL_GT, queryL_GT);
[eva.precision_text2image, eva.recall_text2image] = precision_recall(idx_rank', trainL_GT, queryL_GT);

end

