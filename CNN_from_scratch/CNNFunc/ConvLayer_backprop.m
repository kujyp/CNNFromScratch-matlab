function prev_d = ConvLayer_backprop(d, a, w)
prev_d = zeros([size(d,1), size(d,2), size(w,3)]);
d = d .* a .* (1-a);
pad = padarray(d,[floor(size(w,1)/2),floor(size(w,2)/2)]);
for k3 = 1:size(w,3)
    deconved = convn(pad, squeeze(w(:,:,k3,end:-1:1)),'valid');
    prev_d(:,:,k3) = deconved;
end
% backprop activation func

end