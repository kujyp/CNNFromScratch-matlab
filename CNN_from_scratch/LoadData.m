% function trains = LoadData(varargin)
% %type = 'MNIST';
% %type = 'ANIMAL';
% 
% % initial
% trains = dataset();
% type = 'ANIMAL';
% img_size = [28,28];
% 
% 
% if nargin == 1
%     type = varargin{1};
%     trains = LoadData2(type,img_size);
% elseif nargin == 2
%     type = varargin{1};
%     img_size = varargin{2};
%     trains = LoadData2(type,img_size);
% elseif nargin == 3
%     type= varargin{1};
%     img_size = varargin{2};
%     idx = varargin{3};
%     trains = LoadData2(type,img_size,idx);
% end
% 
% end
% 

function trains = LoadData(type,img_size,varargin)
normalization = 1;
trains=dataset();
switch(type)
    case 'MNIST'
        load('MNIST.mat')
        temp_images = reshape(MNIST.Images,[60000,28,28,1]);
        if size(varargin) == 1
            idx = varargin{1};
            temp_images = temp_images(idx,:,:,:);
        end
        trains.images = zeros(size(temp_images,1),img_size(1),img_size(2),1);
        trains.dec = MNIST.Labels;
        trains.dec = trains.dec+1;
        refmat = eye(max(trains.dec));
        trains.labels = refmat(trains.dec, :);
        trains.labelnames = {'0','1','2','3','4','5','6','7','8','9'};

    case 'ANIMAL'
        load('Data.mat')
        normalization = 1/255;
        temp_images = reshape(Data_proj2.Images, [114,300,300,1]);
        trains.images = zeros(114,img_size(1),img_size(2),1);
        trains.dec = Data_proj2.Labels;
        trains.dec = trains.dec+1;
        refmat = eye(max(trains.dec));
        trains.labels = refmat(trains.dec, :);
        trains.labelnames = {'Elephant','Crocodile'};
end

% save and normalization
for k1 = 1:size(trains.images,1)
    trains.images(k1,:,:) = double(imresize(squeeze(temp_images(k1,:,:,:)), [size(trains.images,2),size(trains.images,3)]))*normalization;
end

end


