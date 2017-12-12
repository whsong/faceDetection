function [windows] = slidingWindow(im, startSize)
%SLIDINGWINDOW return a series of window for the given image
%   startSize [w,h] row vector 
% windows  consists of row vectors [left top width height], currently only consider square region
[h,w] = size(im);
scales = floor(1.2.^[3:4]'.*startSize);
windows = [];
for i = 1:length(scales)
    iScale = scales(i,:);
    stepSize = ceil(iScale./2);
    % windowsPerImage = floor((size(im) - iScale)./stepSize + 1);
    rowInd = 1:stepSize(2):h - iScale(2);
    colInd = 1:stepSize(1):w - iScale(1);
    [C,R] = meshgrid(colInd,rowInd);
    newWindows = [C(:),R(:), repmat(iScale,numel(C),1)];
    windows = [windows;newWindows];
end
end

