function [param, train, query] = load_dataset(param)

if strcmp(param.ds_name, 'Ssense')
    param.nquery = 2000;
    param.chunk_size = 2000;
    param.num_class1 = 4;
    param.num_class2 = 28;
elseif strcmp(param.ds_name, 'FashionVC')
    param.nquery = 2000;
    param.chunk_size = 2000;
    param.num_class1 = 8;
    param.num_class2 = 27;
elseif strcmp(param.ds_name, 'NUSWIDE7-21_deep')
    param.nquery = 5000;
    param.chunk_size = 10000;
    param.num_class1 = 7;
    param.num_class2 = 21;
else
    error('DATASET NAME: ERROR!\n');
end

if strcmp(param.ds_name, 'FashionVC') || strcmp(param.ds_name, 'Ssense')
    load(fullfile(param.ds_dir, [param.ds_name '.mat']));
    X = Gist;       clear Gist
    Y = Tag;        clear Tag
    L = Label;      clear Label
elseif strcmp(param.ds_name, 'NUSWIDE7-21_deep')
    load(fullfile(param.ds_dir, [param.ds_name '.mat']));
    X = Image;      clear Image
    Y = Tag;        clear Tag
    L = Label;      clear Label
end

[param, train, query] = split_dataset(X, Y, L, param);
train.A1_2 = A1_2;
end