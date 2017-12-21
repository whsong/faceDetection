
rootDir = 'D:\Documents\UMass\Study\17Fall\COMPSCI670-SEC01 Computer Vision Fall 2017\Final Project\final project papers\WIDER\';
metaDir = fullfile(rootDir, 'wider_face_split');
val = load(fullfile(metaDir,'wider_face_train.mat'));

% evalDir = fullfile(rootDir, 'eval_tools', 'ground_truth');
% val = load(fullfile(evalDir,'wider_face_val.mat'));
fields = fieldnames(val);
rng(0);
numDir = size(val.(fields{1}),1);
ind = cell(numDir,1);
for i=1:numDir
    numImages = size(val.blur_label_list{i},1);
    ind{i} = randperm(numImages, ceil(numImages*0.1));
end
for i=1:size(fields,1)
    iField = fields{i};
    for j=1:numDir
        if iscell(val.(iField){j})
        val.(iField){j} = val.(iField){j}(ind{j});
        end
    end
end
save(fullfile(metaDir,'wider_face_train_10.mat'),'-struct','val');