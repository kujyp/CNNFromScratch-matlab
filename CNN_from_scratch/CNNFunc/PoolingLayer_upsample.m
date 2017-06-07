function pre_a = PoolingLayer_upsample(a,maxloc,w_size)
pre_a = zeros(size(a).*[w_size,1]);
for w_idx = 1:size(a,3)
    kroned = kron(a(:,:,w_idx),ones(w_size));
    pre_a(:,:,w_idx) = kroned.* maxloc(:,:,w_idx);
end

end
            