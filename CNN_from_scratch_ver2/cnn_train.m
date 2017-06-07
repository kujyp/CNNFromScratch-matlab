function opt_weight = cnn_train(trains, weight, conf)
epochs = conf.epochs;
alpha = conf.alpha;

labels = trains.labels;
images = trains.images;
clear trains.data;
m = length(labels);

it = 0;
for e = 1:epochs
    for it = 1,
        mb.images = images;
        mb.labels = labels;
        
        [cost, grad,~] = cnn_cost(mb, weight);
        cost
        acc = cnn_evaluation(mb, weight);

        weight = cnn_gradientDescent(weight,grad,alpha);
        % velocity = mom * velocity + alpha * grad / 114;
        % theta = theta - velocity;
        
        % alpha = alpha/2.0;
    end
end

opt_weight = weight;

end