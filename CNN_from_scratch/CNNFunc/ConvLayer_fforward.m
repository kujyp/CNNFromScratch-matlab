function next_a = ConvLayer_fforward(a,w,b)

next_a = zeros([size(a,1), size(a,2), size(w,4)]);
pad = padarray(a,[floor(size(w,1)/2),floor(size(w,2)/2)]);
for k4 = 1:size(w,4)
    conved = convn(pad, rot90(w(:,:,end:-1:1,k4),2),'valid');
    next_a(:,:,k4) = sigmoid(conved + b(k4));
end

end