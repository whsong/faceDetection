
% inputDir1 = '..\images\YALE\unpadded\';

% testFiles = dir(fullfile('..','nasa_small.jpg'));
% iFile = testFiles(1);
% imOrigin = imread(fullfile(iFile.folder,iFile.name));
% im = rgb2gray(imOrigin);

faceDetector = vision.CascadeObjectDetector;%Pre-trained square front face detector. can change to other detector.
rootDir = 'D:\Documents\UMass\Study\17Fall\COMPSCI670-SEC01 Computer Vision Fall 2017\Final Project\final project papers\WIDER\';
evalDir = fullfile(rootDir, 'WIDER_val\images\');
evalFiles = dir(fullfile(evalDir, '*', '*.jpg'));
outDir = fullfile(rootDir, 'eval_tools', 'pred');
numImages = length(evalFiles);
% results(numImages) = struct('Boxes',[],'Scores',[]);
for i=1:numImages
    iFile = evalFiles(i);
I = imread(fullfile(iFile.folder,iFile.name));
bboxes = step(faceDetector, I);
scores = ones(size(bboxes,1),1);% Problem is here. Viola-Jones method does have have the concept of score and I made up one.
%     results(i).Boxes = bboxes;
%     results(i).Scores = scores;
% IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, '');
% figure, imshow(IFaces), title('Detected faces');

C = strsplit(iFile.folder,filesep);
outSubFolder = fullfile(outDir, C{end});
if ~exist(outSubFolder, 'dir')
    mkdir(outSubFolder);
end

numFaces = size(bboxes,1);
[~,filename,ext] = fileparts(iFile.name);
txtFile = fullfile(outSubFolder, [filename, '.txt']);
fileID = fopen(txtFile,'w');
fprintf(fileID,'%s\n',filename);
fprintf(fileID,'%d\n',numFaces);
fprintf(fileID,'%d %d %d %d %.2f\n',[bboxes, scores].');
fclose(fileID);
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