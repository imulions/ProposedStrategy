function net = mlp_hu_free1(Inputs, TargetData)
L = 100;
X = Inputs;
N = size(Inputs, 1);
Y = TargetData;
actFun = @(tempH) 2 ./ (1 + exp(-2*tempH))-1;
learning_rate = 0.01;
max_iter = 10000;
w = rand(size(Inputs, 2), L);
tempH = X*w; 
H = actFun(tempH);
clear tempH;
if size(H,1)>=size(H,2)
    outputWeight = pinv(eye(size(H,2))/1000 + H' * H) * H' * Y; 
else
    outputWeight = H' * (pinv(eye(size(H,2))/1000 + H * H') * Y);
end
for iter = 1:max_iter
    Y_pred = H * outputWeight;
    gradient = -2 * H' * (Y - Y_pred);  
    outputWeight = outputWeight - learning_rate * gradient / N;
    if mod(iter, 100) == 0
        loss = sum(sum((Y - Y_pred).^2)) / N;
        disp(['Iteration ', num2str(iter), ' - Loss: ', num2str(loss)]);
    end
end
ielm.outputWeight = outputWeight;
ielm.inputWeight = w;
net = ielm;
save('NetPara.mat', "net")
end





