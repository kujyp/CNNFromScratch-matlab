function weight = cnn_weight_init(conf)
numFilts = length(conf.filterDim);
numPlanes = conf.numPlanes;
filtDim = conf.filterDim;
init = conf.w_init;
weight.w = cell(numFilts,1);
weight.b = cell(numFilts,1);
weight.type = cell(numFilts,1);
for idx = 1:numFilts
    w_siz = [filtDim(idx),filtDim(idx),numPlanes(idx),numPlanes(idx+1)];
    weight.w{idx,1} = init*randn(w_siz);
    weight.b{idx,1} = init*randn(numPlanes(idx+1),1);
    weight.type{idx} = 'conv';
end
end