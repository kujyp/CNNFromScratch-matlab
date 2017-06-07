%clear;
%close all;
addpath(genpath('..'))

%trains = LoadData('MNIST', [28,28]);
trains = LoadData('ANIMAL',[28, 28]);
%trains.images = trains.images(1:1000,:,:);
params.test_range = 1:100;
vis.sampleview_num = 5;

trains.imagesize = [size(trains.images,2),size(trains.images,3),size(trains.images,4)];

%% initialize parameters

cnn.filter_num = [16,32];

%cnn.hidden_neuron_num = {[14,14,32]; [7,7,16]};
%cnn.hidden_neuron_num = {[14,14,32]};

cnn.w_size = {[3,3];[3,3]};
cnn.pool_size = {[2,2];[2,2]};
fnn.hidden_neuron_num = [];
%                  200; 30, ...

cnn.filter_num = cnn.filter_num';
cnn.layer_num = length(cnn.w_size) + length(cnn.pool_size);
cnn.neuron_num = cell(cnn.layer_num+1,1);
cnn.neuron_num{1,1} = trains.imagesize;
cnn.w_num = cell(cnn.layer_num,1);
for k1 = 1:length(cnn.w_size)+length(cnn.pool_size)
    if mod(k1,2) == 1
        cnn.neuron_num{k1+1,1} = [cnn.neuron_num{k1,1}(1:2),cnn.filter_num(ceil(k1/2),1)];
        cnn.w_num{k1,1} = [cnn.w_size{ceil(k1/2),1},cnn.neuron_num{k1,1}(3),cnn.neuron_num{k1+1,1}(3)];
    else
        cnn.neuron_num{k1+1,1} = [floor(cnn.neuron_num{k1,1}(1:2)./cnn.pool_size{k1/2,1}),cnn.neuron_num{k1,1}(3)];
    end
end
%fnn.neuron_num = num2cell([prod(cnn.neuron_num{end});... 
fnn.neuron_num = num2cell([28*28;... 
                  fnn.hidden_neuron_num(:); ...
                  size(trains.labels,2)]);              
fnn.layer_num = length(fnn.neuron_num)-1;

params.learningrate = 0.001;
params.epoch_num = 20000;
params.dropout_rate = 0;
params.initial_w = 1/200;
params.initial_b = 1/200;
%params.normalization_lamda = 0.2;

sigmoid = @(x) 1./(1+exp(-x));
quadraticCost_delta = @(a, y) -(y-a);
choose_one = @(x) x==max(x(:));
choose_max = @(ipt) max(ipt(:));

%% initialize weights
fnn.w = cell(fnn.layer_num, 1);
fnn.b = cell(fnn.layer_num, 1);

for layer_idx = 1:fnn.layer_num,
    row_num = fnn.neuron_num{layer_idx,1};
    col_num = fnn.neuron_num{layer_idx+1,1};
    fnn.w{layer_idx, 1} = randn(row_num, col_num) * params.initial_w;
    fnn.b{layer_idx, 1} = randn(col_num, 1) * params.initial_b;
end

cnn.w = cell(cnn.layer_num,1);
cnn.b = cell(cnn.layer_num,1);
for layer_idx = 1:cnn.layer_num
    if ~isempty(cnn.w_num{layer_idx})
        cnn.w{layer_idx} = randn(cnn.w_num{layer_idx}) / sqrt(prod(cnn.neuron_num{1})) * params.initial_w;
        cnn.b{layer_idx} = randn(cnn.neuron_num{layer_idx+1}(3),1) / sqrt(prod(cnn.neuron_num{1})) * params.initial_b;
    end
end

cnn_v = nn_v;
fnn_v = nn_v;
cnn_v.neuron_value = cell(cnn.layer_num+1, 1);
cnn_v.delta_value = cell(cnn.layer_num+1, 1);
cnn_v.dropout_setting = cell(cnn.layer_num, 1);
cnn_v.maxloc = cell(cnn.layer_num, 1);
fnn_v.dropout_setting = cell(fnn.layer_num, 1);
fnn_v.neuron_value = cell(fnn.layer_num+1, 1);
fnn_v.delta_value = cell(fnn.layer_num+1, 1);

%% Visualize
errorrate_output = [];

h1 = figure('Units','normalized','Position',[0.55 0.45 0.4 0.4]);
for k1=1:vis.sampleview_num^2
    subplot(vis.sampleview_num,vis.sampleview_num,k1);
    imshow(imresize(squeeze(trains.images(params.test_range(k1),:,:,:)),10));
end
drawnow;
h2 = figure('Units','normalized','Position',[0.55 0.05 0.4 0.4]);
rms_v = zeros(2,size(trains.images,1));


%% Training
tic;
for jt = 1:params.epoch_num,
    for it = 1:size(trains.images,1)
        % Feedfoward
        %cnn_v = DropoutSetting(cnn_v, cnn, params.dropout_rate);
        %cnn_v = CNN_FeedFoward(cnn,cnn_v,squeeze(trains.images(it,:,:,:)));
        
        fnn_v = DropoutSetting(fnn_v, fnn, params.dropout_rate);
        fnn_v = FNN_FeedFoward(fnn,fnn_v,reshape(squeeze(trains.images(it,:,:,:)),[numel(squeeze(trains.images(it,:,:,:))) 1]));
        %fnn_v = FNN_FeedFoward(fnn,fnn_v,cnn_v.neuron_value{end,1}(:));
        
        % Output Error
        output_ground_truth = trains.labels(it, :)';
        error = quadraticCost_delta(fnn_v.neuron_value{end,1}, output_ground_truth);

        % Back Propagation
        fnn_v = FNN_Backpropagation(fnn,fnn_v,error);
        %error = reshape(fnn_v.delta_value{1,1},cnn.neuron_num{end});
        %cnn_v = CNN_Backpropagation(cnn,cnn_v,error);
        
        % Output Gradient + Gradient Descent
        fnn = FNN_gradientDescent(fnn, fnn_v, params.learningrate);
        %cnn = CNN_gradientDescent(cnn, cnn_v, params.learningrate);
        %% Training End
        
        %% debug via cost rms
        % Feedfoward
        %cnn_v = CNN_FeedFoward(cnn,cnn_v,squeeze(trains.images(it,:,:,:)));
        %fnn_v = FNN_FeedFoward(fnn,fnn_v,cnn_v.neuron_value{end,1}(:));
        fnn_v = FNN_FeedFoward(fnn,fnn_v,reshape(squeeze(trains.images(it,:,:,:)),[numel(squeeze(trains.images(it,:,:,:))) 1]));
        % Output Error
        output_ground_truth = trains.labels(it, :)';
        error = quadraticCost_delta(fnn_v.neuron_value{end,1}, output_ground_truth);

        rms_v(1,it) = rms(fnn_v.delta_value{end,1});
        rms_v(2,it) = rms(fnn_v.delta_value{end,1}) - rms(error);
        
        if mod(it,500) ==0
            figure(h2);
            title([num2str(it),'/',num2str(size(trains.images,1))]);
            plot(1:it,rms_v(1,1:it),'b'); % #1
            hold on;
            plot(1:it,rms_v(2,1:it),'r'); % #2
            drawnow;
        end
        %% debug end
        
    end
    
    %% Test set
    predict_y = zeros(size(trains.labels(params.test_range,:,:)));
    y_idx = 1;
    for tit = params.test_range
        % Feedfoward
        %cnn_v = CNN_FeedFoward(cnn,cnn_v,squeeze(trains.images(tit,:,:,:)));
        fnn_v = FNN_FeedFoward(fnn,fnn_v,reshape(squeeze(trains.images(it,:,:,:)),[numel(squeeze(trains.images(it,:,:,:))) 1]));
        %fnn_v = FNN_FeedFoward(fnn,fnn_v,cnn_v.neuron_value{end,1}(:));
        % Output Error
        output_ground_truth = trains.labels(it, :)';

        predict_y(y_idx, :) = choose_one(fnn_v.neuron_value{end,1});
        y_idx = y_idx +1;
    end
    error_rate = sum(sum(abs(predict_y - trains.labels(params.test_range,:)))) / (2 * length(params.test_range));
    
    errorrate_output = [errorrate_output; jt, error_rate];    
    disp(['iter :', 2int(jt), 'accuracy : ',, error_rate]
    
    figure(h1);
    for(k1 = 1:vis.sampleview_num^2)
        subplot(vis.sampleview_num,vis.sampleview_num,k1);
        title(find(predict_y(k1,:)==1)-1);
    end
    drawnow;
end
toc;

errorrate_min = min(errorrate_output(:,2));
errorrate_min_index = find(errorrate_output(:,2) == min(errorrate_output(:,2)));

errorrate_output = [errorrate_output; errorrate_min_index(1), errorrate_min(1)];
[errorrate_min_index(1), errorrate_min(1)]
