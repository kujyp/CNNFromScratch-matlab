function shw(varargin)

sz = size(varargin{1}.images);
fidx = 1:sz(1);
if(length(fidx)>25)
    idx = ceil(rand(25,1)*length(fidx));
elseif(length(fidx)>16)
    idx = ceil(rand(16,1)*length(fidx));
elseif(length(fidx)>9)
    idx = ceil(rand(16,1)*length(fidx));
else
    idx = 1;
end
if nargin == 3
    shw3(varargin{1},varargin{2},varargin{3});
elseif nargin == 2
    shw3(varargin{1},varargin{2},idx);
else
    shw3(varargin{1},sz(2:3),idx);
end
end

function shw3(train, siz,idx)
figure;
for k1 = 1:length(idx)
    subplot(ceil(length(idx)^0.5),ceil(length(idx)^0.5),k1);
    imshow(squeeze(reshape(train.images(idx(k1),:,:),siz)));
    %label_idx = find(train.labels(k1)==1)+1;
    label_idx = train.labels(idx(k1))+1;
    title(train.labelnames(label_idx))
end
end

