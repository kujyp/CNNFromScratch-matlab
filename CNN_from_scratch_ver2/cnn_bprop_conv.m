function pre_d = cnn_bprop_conv(d,a,w)
    

w = rot90(w,2);

filterDim = size(w, 1);
numInplanes = size(w, 3);
numOutplanes = size(w, 4);
assert (numInplanes == size(d, 3))

numImages = size(d, 4);
imageDim = size(d, 1);
convDim = imageDim + filterDim - 1;

pre_d = zeros(convDim, convDim, numOutplanes, numImages);

if size(a,1) ~= size(d,1), a = reshape(a,size(d)); end
d = d.*a.*(1-a);
% ---- Convolutions ----
for imageNum = 1:numImages
  for outplane = 1: numOutplanes
    convolvedImage = zeros(convDim, convDim);
    for inplane = 1: numInplanes
      filter = w(:,:,inplane,outplane);
      filter = rot90(squeeze(filter), 2);
      im = squeeze(d(:,:,inplane,imageNum));
      convolvedImageLoop = conv2(im, filter, 'full');
      convolvedImage = convolvedImage + convolvedImageLoop;
    end
    pre_d(:,:,outplane,imageNum) = convolvedImage;
  end
end
end