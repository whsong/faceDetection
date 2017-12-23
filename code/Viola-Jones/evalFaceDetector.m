
% inputDir1 = '..\images\YALE\unpadded\';

% testFiles = dir(fullfile('..','nasa_small.jpg'));
% iFile = testFiles(1);
% imOrigin = imread(fullfile(iFile.folder,iFile.name));
% im = rgb2gray(imOrigin);

% faceDetector = vision.CascadeObjectDetector;%Pre-trained square front face detector. can change to other detector.
faceDetector = rcnn;
rootDir = 'D:\Documents\UMass\Study\17Fall\COMPSCI670-SEC01 Computer Vision Fall 2017\Final Project\final project papers\WIDER\';
metaDir = fullfile(rootDir, 'wider_face_split');
val = load(fullfile(metaDir,'wider_face_val_10.mat'));

evalDir = fullfile(rootDir, 'WIDER_val\images\');
% evalFiles = dir(fullfile(evalDir, '*', '*.jpg'));
outDir = fullfile(rootDir, 'eval_tools', 'pred');
% numImages = length(evalFiles);
% results(numImages) = struct('Boxes',[],'Scores',[]);
fields = fieldnames(val);
numDir = size(val.(fields{1}),1);
for i=1:numDir
    iEvent = val.event_list{i};
    iFileList = val.file_list{i};
    numImages = size(iFileList,1);
    
    for j=1:numImages
        iFilename = fullfile(evalDir, iEvent, [iFileList{j},'.jpg']);
%         iFile = evalFiles(j);
        I = imread(iFilename);
%         I = imread(fullfile(iFile.folder,iFile.name));
        [bboxes,scores] = detect(faceDetector,I);
%         bboxes = step(faceDetector, I);
%         scores = [size(bboxes,1):-1:1].';% Problem is here. Viola-Jones method does have have the concept of score and I made up one.
        %     results(i).Boxes = bboxes;
        %     results(i).Scores = scores;
        IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, '');
        figure, imshow(IFaces), title('Detected faces');
        
%         C = strsplit(iFile.folder,filesep);
%         outSubFolder = fullfile(outDir, C{end});
        outSubFolder = fullfile(outDir, iEvent);
        if ~exist(outSubFolder, 'dir')
            mkdir(outSubFolder);
        end
        numFaces = size(bboxes,1);
        %% write results to .txt file
%         [~,filename,ext] = fileparts(iFile.name);
%         txtFile = fullfile(outSubFolder, [filename, '.txt']);
        txtFilename = fullfile(outSubFolder, [iFileList{j}, '.txt']);
        fileID = fopen(txtFilename,'w');
        fprintf(fileID,'%s\n',iFileList{j});
        fprintf(fileID,'%d\n',numFaces);
        fprintf(fileID,'%d %d %d %d %.2f\n',[bboxes, scores].');
        fclose(fileID);
    end
end
% results = struct2table(results);
% S = load(fullfile(rootDir, 'wider_face_split', 'wider_face_val.mat'));
% trainingData = S_facebbx_list;
%
% [ap,recall,precision] = evaluateDetectionPrecision(results,trainingData);

%%
% % Plot the precision-recall curve.
% figure
% plot(recall,precision)
% grid on
% title(sprintf('Average Precision = %.1f',ap))