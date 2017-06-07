    function cnn = CNN_gradientDescent(cnn, cnn_v, learningrate)
for layer_idx = 1:2:cnn.layer_num
    [grad_w, grad_b] = ConvLayer_backprop_w(cnn_v.delta_value{layer_idx+1,1}, cnn_v.neuron_value{layer_idx,1}, cnn.w_size{ceil(layer_idx/2),1});

    cnn.w{layer_idx, 1} = cnn.w{layer_idx , 1} ...
        - learningrate * grad_w;
    cnn.b{layer_idx, 1} = cnn.b{layer_idx, 1} ...
        - learningrate * grad_b;
end
end