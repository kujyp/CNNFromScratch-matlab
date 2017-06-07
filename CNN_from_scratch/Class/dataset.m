classdef dataset
    properties
        images;
        dec;
        labels;
        labelnames;
    end
    
    methods
        function obj = dataset(varargin)
            type = 'ANIMAL';
            img_size = [28,28];

            if nargin == 0
                return;
            end

            if nargin == 1
                type = varargin{1};
                trains = LoadData(type,img_size);
            elseif nargin == 2
                type = varargin{1};
                img_size = varargin{2};
                trains = LoadData(type,img_size);
            elseif nargin == 3
                type= varargin{1};
                img_size = varargin{2};
                idx = varargin{3};
                trains = LoadData(type,img_size,idx);
            end
            obj = trains;
        end
        
        function shw(varargin)
            sz = size(images);
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
            
            figure;
            for k1 = 1:length(idx)
            subplot(ceil(length(idx)^0.5),ceil(length(idx)^0.5),k1);
            imshow(squeeze(reshape(images(idx(k1),:,:),siz)));
            %label_idx = find(train.labels(k1)==1)+1;
            label_idx = dec(idx(k1));
            title(labelnames(label_idx));
            end
            
%             if nargin == 3
%                 shw3(varargin{1},varargin{2},varargin{3});
%             elseif nargin == 2
%                 shw3(varargin{1},varargin{2},idx);
%             else
%                 shw3(varargin{1},sz(2:3),idx);
%                 %shw3(sz(2:3),idx);
%             end
        end

    end
end