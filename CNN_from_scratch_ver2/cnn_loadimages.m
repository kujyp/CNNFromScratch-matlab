function data = cnn_loadimages(type, siz)
% Input : type(string), siz(2x1 int)
% Output : data.images / data.labels
% There is futher informations in ResultReport

% initial value
if ~exist('type','var') type = 'elephant'; end
rot_flag = false;

addpath(genpath('..'));    
            
    switch(type)
        case 'elephant'
            filename = '/dataset/elephant.mat';
            load(filename);
            images = Data_proj2.Images;
            labels = Data_proj2.Labels; 
            
            images = permute(images,[2 1]);
            images = double(images);
            images = images./255;
            images = reshape(images,300,300,1,[]);
            if exist('siz','var')
                total_siz = [siz(1),siz(2),size(images,3),size(images,4)];
            else
                total_siz = size(images);
            end
            label_names = {'Croco','Eleph'};
            
            
        case 'cifar'
            filename = {'/dataset/cifar-10-batches-mat/data_batch_1.mat',...
                        '/dataset/cifar-10-batches-mat/batches.meta.mat'};
            load(filename{1});
            load(filename{2});
            images = data;
            %labels = labels;
            %label_names = label_names;
            
            images = permute(images, [2 1]);
            images = double(images);
            images = images./255;
            images = reshape(images, 32,32,3,[]);
            if exist('siz','var')
                total_siz = [siz(1),siz(2),size(images,3),size(images,4)];
            else
                total_siz = size(images);
            end
            rot_flag = true;
            
            clear data;
            
        case 'MNIST'
            filename = '/dataset/MNIST.mat';
            load(filename);
            images = MNIST.Images;
            labels = MNIST.Labels;
            %label_names = label_names;
            
            images = permute(images, [2 1]);
            images = reshape(images, 28,28,1,[]);
            if exist('siz','var')
                total_siz = [siz(1),siz(2),size(images,3),size(images,4)];
            else
                total_siz = size(images);
            end
            label_names = {'Zero', 'One', 'Two','Three','Four','Five','Six','Seven','Eight','Nine'};
            clear data;
    end
    
    data.images = zeros(total_siz);
    
    % Image resize
    if exist('siz','var')
        for idx = 1:total_siz(4)
            data.images(:,:,:,idx)  = imresize(images(:,:,:,idx),[total_siz(1),total_siz(2)]);
        end
    else
        data.images = images;
    end
            
    data.labels = labels;
    data.rot_flag = rot_flag;
    data.label_names = label_names;
end