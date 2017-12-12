
%% training
inputDir1 = '..\images\YALE\unpadded\';
inputDir2 = '..\images\att_faces\*\';
faceFiles1 = dir(fullfile(inputDir1, '*.pgm'));
faceFiles2 = dir(fullfile(inputDir2, '*.pgm'));
faceFiles = [faceFiles1;faceFiles2];
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
imOrigin = imread(fullfile(iFile.folder,iFile.name));
im = rgb2gray(imOrigin);
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
conf = score(:,1);
positiveWindow = windows(label==1,:);
% NMS
bbox = windows;
bbox(:,3:4) = bbox(:,3:4)+bbox(:,1:2);

confthresh=0;
indsel=find(conf>confthresh);
[nmsbbox,nmsconf]=prunebboxes(bbox(indsel,:),conf(indsel),0.2);

confthreshnms=0;
nmsbbox=nmsbbox(nmsconf>confthreshnms,:);
[nmsconf,I] = sort(nmsconf,'descend');
nmsbbox = nmsbbox(I,:);
% showbbox(nmsbbox(indsel,:)
nmsWindow = nmsbbox;
nmsWindow(:,3:4) = nmsWindow(:,3:4)-nmsWindow(:,1:2);
%% plot
imshow(imOrigin);
for i=1:20%size(nmsWindow,1)
    iPositiveWindow = nmsWindow(i,:);
    rectangle('Position', iPositiveWindow, 'EdgeColor','g', 'LineWidth',2);
end

function features = extractFeatures(im)

    im = imresize(im,[112,92]);% the size of att_faces, TODO
%     im = sqrt(double(im));
    % N = prod([BlocksPerImage, BlockSize, NumBins]), BlocksPerImage = floor((size(I)./CellSize - BlockSize)./(BlockSize - BlockOverlap) + 1)
    features = extractHOGFeatures(im,'CellSize',[8,8], 'BlockSize',[2,2], 'UseSignedOrientation',false, 'NumBins',9);
%     features = extractLBPFeatures(im);
    % Square-root scaling
%     features = sqrt(features);
    % L2-normalization
%     features = features./sqrt(sum(features.^2,1));
end