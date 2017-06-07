function cost = cnn_costSoftmax(probs, labels)
numImages = size(probs,2);

probsL = log(probs);
labels = labels+1;
I = sub2ind(size(probs), reshape(labels, 1, []), 1:numImages);
probsL = probsL(I);
cost = -sum(probsL);

end