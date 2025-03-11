function [y] = mlpfwd(net1, gmm, x)
posterior_probs = gmm.posterior(x);
posterior_probs = posterior_probs ./ sum(posterior_probs, 2);   
tempH=posterior_probs*net1.inputWeight; 
H = 2 ./ (1 + exp(-2*tempH))-1;
y = net1.outputWeight'*H';
end
