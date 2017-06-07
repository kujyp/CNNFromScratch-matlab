function cnn_v = CNN_Backpropagation(cnn,cnn_v,error)

cnn_v.delta_value{cnn.layer_num+1,1} = error;
for layer_idx = cnn.layer_num : -1 : 1
    if mod(layer_idx,2) == 0
        %% upsampling
        cnn_v.delta_value{layer_idx,1} = PoolingLayer_upsample(cnn_v.delta_value{layer_idx+1,1}, ...
                                        cnn_v.maxloc{layer_idx,1},cnn.pool_size{layer_idx/2,1});
    else
        cnn_v.delta_value{layer_idx,1} = ConvLayer_backprop(cnn_v.delta_value{layer_idx+1,1},...
                                                        cnn_v.neuron_value{layer_idx+1,1},cnn.w{layer_idx});
    end
end
end
