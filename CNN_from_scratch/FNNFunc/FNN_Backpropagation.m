function fnn_v = FNN_Backpropagation(fnn,fnn_v,error)
fnn_v.delta_value{end,1} = error;

for layer_idx = fnn.layer_num : -1 : 1
    if 1 < layer_idx < fnn.layer_num
        fnn_v.delta_value{layer_idx+1, 1} = fnn_v.delta_value{layer_idx+1, 1} .* fnn_v.dropout_setting{layer_idx, 1};
    end

    first_term = fnn.w{layer_idx, 1} ...
        * fnn_v.delta_value{layer_idx + 1, 1};
    second_term = fnn_v.neuron_value{layer_idx , 1} .* ...
        (1 - fnn_v.neuron_value{layer_idx, 1});
    fnn_v.delta_value{layer_idx, 1} = first_term .* second_term;
end

end