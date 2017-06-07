function fnn_v = FNN_FeedFoward(fnn,fnn_v,img)

%% FNN

fnn_v.neuron_value{1,1} = img;

for layer_idx = 1:fnn.layer_num
    output_data_z = fnn.w{layer_idx, 1}' ...
        * fnn_v.neuron_value{layer_idx,1} + fnn.b{layer_idx, 1};

    if layer_idx < fnn.layer_num
        output_data_z = output_data_z .* fnn_v.dropout_setting{layer_idx, 1};
    end

    output_data_a = sigmoid(output_data_z);
    fnn_v.neuron_value{layer_idx+1,1} = output_data_a;
end

end
