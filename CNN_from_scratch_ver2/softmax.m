function probs = softmax(a)
numClasses = size(a,1);
numImages = size(a,2);

probs = zeros(numClasses,numImages);
probs = exp(a);
sumProbs = sum(probs, 1);
probs = bsxfun(@times, probs, 1 ./ sumProbs);
end