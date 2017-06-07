function cnn_massimshow(data,n,idx)

if ~exist('n','var')
    n = 3;
end
if ~exist('idx','var')
    idx = ceil(rand(n,n)*(size(data.images,4)-1))+1;    
end

if size(data.images,4)<n^2
    error('the Number of images is less than n^2');
    return;
end

for img_idx = 1:n^2
    subplot(n,n,img_idx);
    imshowjyp(data,idx(img_idx));
end

end