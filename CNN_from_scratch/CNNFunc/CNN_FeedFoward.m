function cnn_v = CNN_FeedFoward(cnn,cnn_v,img)

cnn_v.neuron_value{1} = img;
        
%% Feedfoward
%% CNN
for layer_idx = 1:cnn.layer_num
    if mod(layer_idx,2) == 1
    % Convoltion
    cnn_v.neuron_value{layer_idx+1,1} = ConvLayer_fforward(cnn_v.neuron_value{layer_idx,1},cnn.w{layer_idx,1},cnn.b{layer_idx,1});
    else
    % Pooling
    [cnn_v.neuron_value{layer_idx+1,1}, cnn_v.maxloc{layer_idx,1}] = PoolingLayer_downsample(cnn_v.neuron_value{layer_idx,1},cnn.pool_size{layer_idx/2,1});
    end
end

end