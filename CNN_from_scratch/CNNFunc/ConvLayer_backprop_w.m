function [grad_w grad_b] = ConvLayer_backprop_w(d,a, w_size)
grad_w = zeros([w_size, size(a,3), size(d,3)]);
grad_b = zeros(size(d,3),1);

pad = padarray(a,floor(w_size/2));
for k4 = 1:size(d,3)
    for k3 = 1:size(a,3)
        grad_w(:,:,k3,k4) = conv2(pad(:,:,k3),rot90(d(:,:,k4),2),'valid');
    end
    grad_b(k4) = sum(sum(d(:,:,k4)));
end
end