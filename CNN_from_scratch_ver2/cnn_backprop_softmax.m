function d = cnn_backprop_softmax(probs,labels)
labels = labels +1;
numImages = size(probs,2);
t = zeros(size(probs));
I = sub2ind(size(probs), reshape(labels, 1, []), 1:numImages);
t(I) = 1;
t = t - probs;
d = -t;

end