function [next_a maxloc] = PoolingLayer_downsample(a,w_size)
choose_max = @(x) max(x(:));
maxloc = zeros(size(a));
next_a = zeros([floor(size(a,1)/w_size(1)), floor(size(a,2)/w_size(2)), size(a,3)]);
for k3 = 1:size(next_a,3)
    for k1 = 1:size(next_a,1)
        for k2 = 1:size(next_a,2)
            next_a(k1,k2,k3) = choose_max(a(k1*w_size(1)-w_size(1)+1:k1*w_size(1),k2*w_size(2)-w_size(2)+1:k2*w_size(2),k3));
            maxloc(k1*w_size(1)-w_size(1)+1:k1*w_size(1),k2*w_size(2)-w_size(2)+1:k2*w_size(2),k3) = a(k1*w_size(1)-w_size(1)+1:k1*w_size(1),k2*w_size(2)-w_size(2)+1:k2*w_size(2),k3) == next_a(k1,k2,k3);
        end
    end
end

end