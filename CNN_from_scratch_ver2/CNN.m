close all;
%clear;

%trains = cnn_loadimages('MNIST');
n=4;
trains.images = ones(n,n,1,2);
trains.images(:,:,:,1) = ones(n,n,1);
trains.images(:,:,:,2) = zeros(n,n,1);
trains.labels = [1 0];
%trains.images(:,:,:,1:10) = rand(n,n,1,10)*0.2;
%trains.images(:,:,:,11:20) = rand(n,n,1,10)*0.2+0.2;
%trains.images(:,:,:,21:30) = rand(n,n,1,10)*0.2+0.4;
%trains.images(:,:,:,31:40) = rand(n,n,1,10)*0.2+0.6;
%trains.labels = ceil((1:40)/10)-1;

%trains = cnn_loadimages('cifar');
%trains = cnn_loadimages('elephant');

% Configuration
conf.numClasses = 2;
conf.imageDim = [n];
%conf.imageDim1 = ;
%conf.imageDim2 = ;
conf.filterDim = [3];
%conf.filterDim = [3 3 3];
conf.numPlanes = [1 1]; 
%conf.numPlane = [1 20 10];
%conf.poolDim = [2 2 2];
conf.w_init = 0.1;

% initial weight
weight = cnn_weight_init(conf);
%weight = 1;

% Learn Parameters
conf.epochs = 20000;
conf.alpha = 1e-1;
conf.momentun = 0;

%% Learning
opt_weight = cnn_train(trains, weight, conf);

%% Test
tests = trains;
acc = cnn_evaluation(tests, opt_weight);

%% Deconv Network
