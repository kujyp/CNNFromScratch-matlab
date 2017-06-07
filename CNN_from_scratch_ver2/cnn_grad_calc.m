function grad = cnn_grad_calc(d,a,type)
switch(type)
    case 'conv'
        grad = cnn_grad_calc_conv(d,a);
    case 'pool'
        pre_d = cnn_bprop_pool(a,w,b);
    case 'dense'
        pre_d = cnn_bprop_dense(a,w,b);
end
end