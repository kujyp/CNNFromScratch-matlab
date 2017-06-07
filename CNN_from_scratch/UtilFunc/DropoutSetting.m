function nn_v = DropoutSetting(nn_v, nn, dropout_rate)

if dropout_rate>0
    for layer_idx = 1:length(nn_v.dropout_setting)
        nn_v.dropout_setting{layer_idx, 1} = (rand([nn.neuron_num{layer_idx+1,1},1]) > dropout_rate);
    end
else
    for layer_idx = 1:length(nn_v.dropout_setting)
        nn_v.dropout_setting{layer_idx, 1} = ones([nn.neuron_num{layer_idx+1,1},1]);
    end
end
end