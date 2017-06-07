function cnn_imshowjyp(data, idx)
if ~exist('idx','var')
    idx = ceil(rand()*(size(data.images,4)-1))+1;    
end
img = data.images(:,:,:,idx);
lab = data.labels(idx);
labname = data.label_names{lab+1};

% resize
if(size(img,1) < 150) img = imresize(img,(150/size(img,1))); end
if(data.rot_flag)
    img = imrotate(img,-90);
end

imshow(img);
title(labname);
end