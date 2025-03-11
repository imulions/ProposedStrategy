close all
clear
clc
load PathS.mat
load pathBeta.mat
dt = 0.001;
tol_cutting = 0;
options.energy='on';
options.suf='off';
options.max_iter=1000;
options.alpha=0;
options.tol_stopping=10^-10;         
options.display = 1;                                     
options.alpha=0.05;         
start_time_train=cputime;
rng(2)
demos = {};
demos{1,1} = pathS(:,1:2)';
demos{1,2} = pathS1(:,1:2)';
[x0, xT, Data, index] = preprocess_demos(demos,dt,tol_cutting);
nin=size(Data,1)/2;
Inputs=Data(1:nin,:)';
TargetData=Data(nin+1:end,:)';
X1 = Inputs(1:701, :) - Inputs(702:end, :);
Y1 = TargetData(1:701, :) - TargetData(702:end, :);
demos1 = {};
demos1{1,1} = pathBeta(:,1:2)';
demos1{1,2} = pathBeta1(:,1:2)';
[x0, xT, Data1, index] = preprocess_demos(demos1,dt,tol_cutting);
x_exp = Data1(1:2, 1: 1201);
v_exp = Data1(3:4, 1: 1201);
nin=size(Data1,1)/2;
Inputs=Data1(1:nin,:)';
TargetData=Data1(nin+1:end,:)';
X2 = Inputs(1:1201, :) - Inputs(1202:end, :);
Y2 = TargetData(1:1201, :) - TargetData(1202:end, :);
K = 20;
X = [X1; X2];
% options = statset('MaxIter', 1000, 'TolFun', 1e-6);
% gmm = fitgmdist(X, K, 'CovarianceType', 'full', 'Options', options);
% posterior_probs = gmm.posterior(X);
% posterior_probs = posterior_probs ./ sum(posterior_probs, 2); 
% save('gmmPara.mat', 'gmm')
%%
Y_all = [Y1; Y2];
% net=mlp_hu_free1(posterior_probs, Y_all);
load gmmPara.mat
load NetPara.mat
%%
end_time_train=cputime;
end_time_train=end_time_train-start_time_train; 
opt_sim.dt = 0.001;
opt_sim.i_max = 100000;
opt_sim.tol = 0;
x0_all = x_exp(1:2, 1);
fn_handle = @(x) mlpfwd(net, gmm, x'); 
[x xd t] = Simulation(x0_all,[],fn_handle,v_exp,x_exp,opt_sim);
x_all=[];
for i=1:size(x,3)
    x_all=[x_all x(:,:,i)];
end
e = [];
for i = 1 : size(x, 3)
    e(:, :, i) = abs(x(:,:,i) - x_exp(:, :, i));
end
figure(1)
hold on
plot(x_exp(1,:),x_exp(2,:),'r.','linewidth',1.5);
n=size(x_all,2)/size(x,3);
for i=1:size(x,3)
    plot(x_all(1,n*(i-1)+1:n*i),x_all(2,n*(i-1)+1:n*i),'k','linewidth',2)
end
set(gca,'xtick',[],'xticklabel',[]);
set(gca,'ytick',[],'yticklabel',[])
