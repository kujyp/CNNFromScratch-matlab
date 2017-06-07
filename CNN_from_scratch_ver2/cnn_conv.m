function next_a = cnn_conv(images,W,b)
    
filterDim = size(W, 1);
numInplanes = size(W, 3);
numOutplanes = size(W, 4);
assert (numInplanes == size(images, 3))

numImages = size(images, 4);
imageDim = size(images, 1);
convDim = imageDim - filterDim + 1;

next_a = zeros(convDim, convDim, numOutplanes, numImages);

% ---- Convolutions ----
for imageNum = 1:numImages
  for outplane = 1: numOutplanes
    convolvedImage = zeros(convDim, convDim);
    for inplane = 1: numInplanes
      filter = W(:,:,inplane,outplane);
      filter = rot90(squeeze(filter), 2);
      im = squeeze(images(:,:,inplane,imageNum));
      convolvedImageLoop = conv2(im, filter, 'valid') + b(outplane);
      convolvedImage = convolvedImage + convolvedImageLoop;
    end
    % Summation across input planes is prior to being activated!
    convolvedImage = 1 ./ (1+exp(-convolvedImage)); 
    next_a(:,:,outplane,imageNum) = convolvedImage;
  end
end
end