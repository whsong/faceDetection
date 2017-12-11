
%% training
inputDir = 'YALE\unpadded\';
faceFiles = dir(fullfile(inputDir, '*.pgm'));
bgFiles = dir(fullfile('bground','*.jpg'));
nFaceFiles = length(faceFiles);
nBgFiles = length(bgFiles);
files = [faceFiles;bgFiles];
features=[];
for i=1:nFaceFiles+nBgFiles
    iFile = files(i);
    im = imread(fullfile(iFile.folder,iFile.name));
    features(i,:) = extractFeatures(im);
    
end
labels = nan(nFaceFiles+nBgFiles,1);
labels(1:nFaceFiles) = 1;
labels(nFaceFiles+1:end) = 0;

SVMModel = fitcsvm(features,labels,'kernelFunction','linear','kernelScale',1,'ClassNames',[1,0]);%'rbf'
%% testing
testFiles = dir(fullfile('nasa_small.jpg'));
iFile = testFiles(1);
im = rgb2gray(imread(fullfile(iFile.folder,iFile.name)));
windows = slidingWindow(im, [20,20]);

nWindow = size(windows,1);
label = nan(nWindow,1);
score = nan(nWindow,2);
for i=1:nWindow
    iWindow = windows(i,:);
    imWindow = im(iWindow(2):iWindow(2)+iWindow(4), iWindow(1):iWindow(1)+iWindow(3));
    testFeatures = extractFeatures(imWindow);
    [label(i),score(i,:)] = predict(SVMModel,testFeatures);
end

positiveWindow = windows(label==1,:);
%% plot
imshow(im);
for i=1:size(positiveWindow,1)
    iPositiveWindow = positiveWindow(i,:);
    rectangle('Position', iPositiveWindow, 'EdgeColor','g', 'LineWidth',2);
end
function features = extractFeatures(im)

    im = imresize(im,[112,92]);% the size of att_faces, TODO
    % N = prod([BlocksPerImage, BlockSize, NumBins]), BlocksPerImage = floor((size(I)./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1)
    features = extractHOGFeatures(im,'CellSize',[8,8], 'BlockSize',[2,2], 'UseSignedOrientation',false, 'NumBins',8);
end