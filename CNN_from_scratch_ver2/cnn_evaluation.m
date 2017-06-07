function acc = cnn_evaluation(tests, weight)
[~, ~, preds] = cnn_cost(tests, weight);
acc = sum(preds==(tests.labels+1))/length(preds);

fprintf('Accuracy is %f\n',acc);

end