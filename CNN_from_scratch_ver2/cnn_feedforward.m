function next_a = cnn_feedforward(a, w, b, type)
switch(type)
    case 'conv'
        next_a = cnn_conv(a,w,b);
    case 'pool'
        next_a = cnn_pool(a,w,b);
    case 'dense'
        next_a = cnn_dense(a,w,b);
end
end