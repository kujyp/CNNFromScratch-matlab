%clear;
%load fnn
%load fnn_v
% idx = 3;
% ipt =fnn_v.neuron_value{idx,1};
% opt =fnn_v.neuron_value{idx+1,1};
% w = fnn.w{idx,1};
%b = fnn.b{idx,1};

%sum(sum( sigmoid((ipt'*w)'+b) ~= opt))

idx = 4;
ipt = fnn_v.delta_value{idx,1};
opt = fnn_v.delta_value{idx-1,1};
w = fnn.w{idx-1,1};
a = fnn_v.neuron_value{idx,1};
z = ipt.*a.*(1-a);
pre_a = (z'*w')';
sum(sum(pre_a ~= opt))


grad_w = fnn_v.neuron_value{idx-1,1} * fnn_v.delta_value{idx, 1}';
grad_b = fnn_v.delta_value{idx, 1};

w = fnn.w{idx-1, 1} ...
        - 0.3 * grad_w;
%fnn.b{layer_idx, 1} = fnn.b{layer_idx, 1} ...
%    - learningrate * grad_b;