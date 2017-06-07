function [cost, grad, preds] = cnn_cost(trains, weight)

images = trains.images;
labels = trains.labels;

w = weight.w;
b = weight.b;
type = weight.type;

numLayers = length(weight.w);
numImages = size(images,4);
a = cell(numLayers+1,1);
d = cell(numLayers+1,1);
grad = cell(numLayers,1);

a{1,1} = images;
for idx = 2:numLayers + 1
%    a{idx,1} = zeros(size(weight.w));
%    d{idx,1} = zeros(size(weight.w));
end

% feedfoward
for idx = 1:numLayers
    a{idx+1,1} = cnn_feedforward(a{idx,1},w{idx,1},b{idx,1},type{idx,1});
end

a{numLayers+1,1} = reshape(a{numLayers+1,1},[],numImages);
%% softmax layer
probs = softmax(a{numLayers+1,1});

% Output Error
cost = cnn_costSoftmax(probs,labels);

%%% predict
    [~,preds] = max(probs,[],1);
    %preds = preds';
    %return;
%end

% softmax layer
d{numLayers+1,1} = cnn_backprop_softmax(probs,labels);
numFilts = size(w{end,1},3);
siz = (numel(d{numLayers+1,1})/numImages/numFilts).^0.5;
d{numLayers+1,1} = reshape(d{numLayers+1,1},siz,siz,numFilts,numImages);

% Back Propagation
for idx = numLayers:-1:1
    d{idx,1} = cnn_backprop(d{idx+1,1},a{idx+1,1},w{idx,1},type{idx,1});
end

%%% Calculate gradient
for idx = 1:numLayers
    %grad = cnn_calculate
end

%% ---------- Gradient Calculation ----------
for idx = numLayers:-1:1
    grad{idx,1} = cnn_grad_calc(d{idx+1,1},a{idx,1},type{idx,1});
end
% softmax layer
%Wd_grad_ = zeros(numel(Wd), numImages);
%for i = 1:numImages
%    Wd_grad_(:, i) = -kron(t(:, i), activationsPooled(:, i));
%end
%Wd_grad = reshape(sum(Wd_grad_, 2), size(Wd'));
%Wd_grad = Wd_grad';
%bd_grad = sum(Dd, 2);

end