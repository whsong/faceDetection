rootDir = 'D:\Documents\UMass\Study\17Fall\COMPSCI670-SEC01 Computer Vision Fall 2017\Final Project\final project papers\WIDER\';
metaDir = fullfile(rootDir, 'wider_face_split');

if ~exist('trainTable300.mat','file')
    %% organize training data
    train = load(fullfile(metaDir,'wider_face_train_10.mat'));
    
    trainDir = fullfile(rootDir, 'WIDER_train', 'images');
    fields = fieldnames(train);
    numDir = size(train.(fields{1}),1);
    trainTable = struct('imageFilename',{},'face',{});% empty struct
    for i=1:numDir
        iEvent = train.event_list{i};
        iFileList = train.file_list{i};
        numImages = size(iFileList,1);
        
        for j=1:numImages
            if mod(j,30)==1
            iFilefullname = fullfile(trainDir, iEvent, [iFileList{j},'.jpg']);
            %         iFile = evalFiles(j);
            gt_boxes = double(train.face_bbx_list{i}{j});
            trainTable(end+1).imageFilename = iFilefullname;
            trainTable(end).face = gt_boxes;
            end
        end
    end
    trainTable = struct2table(trainTable);
    save('trainTable300','trainTable','-v7.3');
else
    load('trainTable300.mat');
end
%% build network
objectClasses = {'face'};
numClassesPlusBackground = numel(objectClasses) + 1;
layers = [ ...
    imageInputLayer([28 28 1])
    convolution2dLayer([5 5],20)
    maxPooling2dLayer([2,2]);
    convolution2dLayer([5 5],50);
    maxPooling2dLayer([2,2]);
%     convolution2dLayer([4 4],500);
    fullyConnectedLayer(numClassesPlusBackground);
    softmaxLayer();
    classificationLayer()];
%% set options
options = trainingOptions('sgdm', ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-6, ...
    'MaxEpochs', 10);
%% train
% acf = trainACFObjectDetector(trainTable,'NumStages',5);
rcnn = trainRCNNObjectDetector(trainTable, layers, options);
% rcnn = trainFastRCNNObjectDetector(trainTable, layers, options);
