testFiles = dir(fullfile('..', 'simple_test.jpg'));
iFile = testFiles(1);
% imOrigin = imread(fullfile(iFile.folder,iFile.name));
% im = rgb2gray(imOrigin);
faceDetector = vision.CascadeObjectDetector;%Pre-trained square front face detector. can change to other detector.

I = imread(fullfile(iFile.folder,iFile.name));
bboxes = step(faceDetector, I);
IFaces = insertObjectAnnotation(I, 'rectangle', bboxes, '');
figure, imshow(IFaces);
% title('Detected faces');
imwrite(IFaces,'simple_test.jpg')