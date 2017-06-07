function pre_d = cnn_backprop(d,a,w,type)
switch(type)
    case 'conv'
        pre_d = cnn_bprop_conv(d,a,w);
    case 'pool'
        pre_d = cnn_bprop_pool(a,w);
    case 'dense'
        pre_d = cnn_bprop_dense(a,w);
end
end