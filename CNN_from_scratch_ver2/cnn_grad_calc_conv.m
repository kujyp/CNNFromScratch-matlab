function grad = cnn_grad_calc_conv(d,a)
numImages = size(d,4);
numOutplane = size(d,3);
numInplane = size(a,3);
filtSiz = size(a,1)-size(d,1)+1;
grad.w = zeros(filtSiz,filtSiz,numInplane,numOutplane);
grad.b = zeros(numOutplane,1);
% second convolutional layer
for i = 1: numImages
    for j = 1: numOutplane
        dR = rot90(squeeze(d(:,:,j,i)), 2);
        for k = 1: numInplane
            im = squeeze(a(:,:,k,i));
            grad.w(:,:,k,j) = grad.w(:,:,k,j) + conv2(im, dR, 'valid');
            grad.b(j) = grad.b(j) + sum(sum(squeeze(d(:,:,j,i))));
        end
    end
end
end