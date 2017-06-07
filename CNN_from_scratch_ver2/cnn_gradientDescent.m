function weight = cnn_gradientDescent(weight,grad,alpha)
for idx = 1:length(grad)
    weight.w{idx,1} = weight.w{idx,1} - grad{idx,1}.w*alpha;
    weight.b{idx,1} = weight.b{idx,1} - grad{idx,1}.b*alpha;
end
end