function [param,train,query] = load_dataset(param)

param.nquery=2000;
param.chunk_size=2000;

if strcmp(param.ds_name,'Ssense')
    param.num_class1=4;
    param.num_class2=28;
elseif strcmp(param.ds_name,'FashionVC')
    param.num_class1=8;
    param.num_class2=27;
else
    error('DATASET NAME: ERROR!\n');
end


fprintf('LOAD DATASET: %s\n',param.ds_name);
if strcmp(param.ds_name,'FashionVC') || strcmp(param.ds_name,'Ssense')
    load(fullfile(param.ds_dir,[param.ds_name '.mat']));
    X=Gist;     clear Gist
    Y=Tag;      clear Tag
    L=Label;    clear Label
end

[param, train, query] = split_dataset(X,Y,L,param);
train.A1_2=A1_2;
end