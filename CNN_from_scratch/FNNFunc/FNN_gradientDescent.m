function fnn = FNN_gradientDescent(fnn, fnn_v, learningrate)
for layer_idx = 1:fnn.layer_num
    grad_w = fnn_v.neuron_value{layer_idx,1} * fnn_v.delta_value{layer_idx+1, 1}';
    grad_b = fnn_v.delta_value{layer_idx+1, 1};

    fnn.w{layer_idx, 1} = fnn.w{layer_idx , 1} ...
        - learningrate * grad_w;
    fnn.b{layer_idx, 1} = fnn.b{layer_idx, 1} ...
        - learningrate * grad_b;
end
end