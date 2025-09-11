%_________________________________________________________________________%
%KELM回归问题       %
%_________________________________________________________________________%
clear;
warning off;
clc
addpath('./MOCGO');%.当前路径
addpath('./CEEMDAN');%.当前路径
%% 导入数据
tic
load data_b;
WA=data;
Nstd=0.5;
NR=200;
MaxIter=5000;
SNRFlag=1;
[modes its]=iceemdan(WA,Nstd,NR,MaxIter,SNRFlag);
WAA=sum(modes(2:end,:));
%% training set
tr=1200
WA1=WAA(1:tr);
m=6
for i=1:m
    AA1(i,:)=WA1(i:end-m+i);
end
P_train=AA1(1:m-1,:);
T_train=AA1(m,:);
%归一化
[Pn_train,inputps]=mapminmax(P_train,-1,1);
[Tn_train,outputps] = mapminmax(T_train,-1,1);
%% testing set
WA2=WAA(tr+1:end);
for i=1:m
    AA2(i,:)=WA2(i:end-m+i);
end
P_test=AA2(1:m-1,:);
T_test=WA(m+tr:end)';
Pn_test = mapminmax('apply',P_test,inputps);
%% 利用基础KELM进行预测
% Regularization_coefficient1 = 4000;  %正则系数
% Kernel_para1 = [200,100,0.5,500,0.4,0.3,0.3];                   %核函数参数矩阵
Kernel_type = 'hyb';
%% MOGWO
dim=5;
nVar=dim;
lb=[10^(-3) 10^(-3) 10^(-3) 1 0];           %下限 正则化系数1 高斯核函数1 poly核函数2  权重1
ub=[10^3 10^3 10^3 10 1];           %上限
Seed_Number=50;       % 种群规模
MaxIt=10;               % Maximum Number of Iterations
Archive_size=50;        % Repository Size
%% MOCGO Parameters
alpha=0.1;  % Grid Inflation Parameter
nGrid=30;   % Number of Grids per each Dimension
beta=4; %=4;    % Leader Selection Pressure Parameter
gamma=2;    % Extra (to be deleted) Repository Member Selection Pressure% Initialization 初始化
%% Initialization
CGO=CreateEmptyParticle(Seed_Number);
Seed=zeros(Seed_Number,nVar);
for i=1:Seed_Number
    CGO(i).Velocity=0;
    CGO(i).Position=zeros(1,nVar);
    for j=1:nVar
        CGO(i).Position(1,j)=unifrnd(lb(j),ub(j),1);
    end
    CGO(i).Cost=fobj(CGO(i).Position',Pn_train,Tn_train,Kernel_type)';
    Seed(i,:)=CGO(i,:).Position;
    CGO(i).Best.Position=CGO(i).Position;
    CGO(i).Best.Cost=CGO(i).Cost;
end
CGO=DetermineDominations(CGO);
Archive=GetNonDominatedParticles(CGO);
Archive_costs=GetCosts(Archive);
G=CreateHypercubes(Archive_costs,nGrid,alpha);
for i=1:numel(Archive)
    [Archive(i).GridIndex Archive(i).GridSubIndex]=GetGridIndex(Archive(i),G);
end
%% Search Process of the CGO
for Iter=1:MaxIt
    for i=0:Seed_Number
        %% Generate New Solutions
        Leader=SelectLeader(Archive,beta);
        % Random Numbers
        I=randi([1,2],1,12); % For Beta and Gamma
        Ir=randi([0,1],1,5); % For Alpha
        % Random Groups
        RandGroupNumber=randperm(Seed_Number,1);
        RandGroup=randperm(Seed_Number,RandGroupNumber);
        % Mean of Random Group
        MeanGroup=mean(Seed(RandGroup,:)).*(length(RandGroup)~=1)...
            +Seed(RandGroup(1,1),:)*(length(RandGroup)==1);
        % New Seeds
        Alpha(1,:)=rand(1,nVar);
        Alpha(2,:)= 2*rand(1,nVar)-1;
        Alpha(3,:)= (Ir(1)*rand(1,nVar)+1);
        Alpha(4,:)= (Ir(2)*rand(1,nVar)+(~Ir(2)));
        ii=randi([1,4],1,3);
        SelectedAlpha= Alpha(ii,:);
        CGO(4*i+1,:).Position=SelectedAlpha(1,:).*(I(1)*Leader.Position-I(2)*MeanGroup);
        CGO(4*i+2,:).Position=Leader.Position+SelectedAlpha(2,:).*(I(3)*MeanGroup-I(4));
        CGO(4*i+3,:).Position=MeanGroup+SelectedAlpha(3,:).*(I(5)*Leader.Position-I(6));
        CGO(4*i+4,:).Position=unifrnd(lb,ub);
        for j=1:4
            % Checking/Updating the boundary limits for Seeds
            CGO(4*i+j,:).Position=min(max(CGO(4*i+j).Position,lb),ub);
            % Evaluating New Solutions
            CGO(4*i+j,:).Cost=fobj(CGO(4*i+j,:).Position',Pn_train,Tn_train,Kernel_type)';
        end
    end
    CGO=DetermineDominations(CGO);
    non_dominated_CGO=GetNonDominatedParticles(CGO);
    Archive=[Archive
        non_dominated_CGO];
    Archive=DetermineDominations(Archive);
    Archive=GetNonDominatedParticles(Archive);
    for i=1:numel(Archive)
        [Archive(i).GridIndex Archive(i).GridSubIndex]=GetGridIndex(Archive(i),G);
    end
    if numel(Archive)>Archive_size
        EXTRA=numel(Archive)-Archive_size;
        Archive=DeleteFromRep(Archive,EXTRA,gamma);
        Archive_costs=GetCosts(Archive);
        G=CreateHypercubes(Archive_costs,nGrid,alpha);
    end
    disp(['In iteration ' num2str(Iter) ': Number of solutions in the archive = ' num2str(numel(Archive))]);
    save results
    costs=GetCosts(CGO);
    Archive_costs=GetCosts(Archive);
end
%% 用优化算法
Regularization_coefficient1=Leader.Position(1);
Kernel_para1=Leader.Position(2:end);
%% 训练
[TrainOutT1,OutputWeight1] = kelmTrain(Pn_train,Tn_train,Regularization_coefficient1,Kernel_type,Kernel_para1);
%% 预测
InputWeight1 = OutputWeight1;
[TestOutT1] = kelmPredict(Pn_train,InputWeight1,Kernel_type,Kernel_para1,Pn_test);
%% 训练集正确率
TrainOutT1 = mapminmax('reverse',TrainOutT1,outputps);%反归一化
errorTrain1 = TrainOutT1 - T_train;
MSEErrorTrain1 = mse(errorTrain1);
%% 测试集正确率
TestOutT1 = mapminmax('reverse',TestOutT1,outputps);
errorTest1 = TestOutT1 - T_test;
MSEErrorTest1 = mse(errorTest1);

%% 测试集准确性
T_sim=TestOutT1;
MAE=mean(abs(T_test-T_sim));
RMSE=sqrt(mean((T_test-T_sim).^2));
MAPE=mean(abs((T_test-T_sim)./T_test));
R2 = corrcoef(T_sim,T_test);
R2 = R2(1,2)^ 2;
[MAE RMSE MAPE R2]

%% 测试集准确性
True=T_test;
PRE=TestOutT1;
Error=TestOutT1-True;
mw=1
AE(mw)=mean(Error);
MAPE(mw)=mean(abs(Error)./True)*100;
MAE(mw)=mean(abs(Error));
RMSE(mw)=sqrt(mean(Error.^2));
NMSE(mw)=mean(((Error).^2)./(True.*PRE));
MdAPE(mw)=median(abs(Error)./abs(True));
FB(mw)=2*(mean(True)-mean(PRE))/(mean(True)+mean(PRE));
R(mw)=myPearson(True,PRE);% 皮尔森相关系数
for iii=2:length(True)
    w(iii-1)=[PRE(iii)- True(iii-1)]*[True(iii)-True(iii-1)];
end
DA(mw)=length(find((w>0)))/length(w);
IA(mw)=1-sum((abs(Error)).^2)/sum((abs(PRE-mean(True))+abs(True-mean(True))).^2);
U1_one=RMSE/(sqrt(mean(True.^2))+sqrt(mean(PRE.^2)));
U2_one=sqrt(mean(((PRE(2:end)-True(2:end))./True(1:end-1)).^2))/sqrt(mean(((PRE(1:end-1)-True(2:end))./True(1:end-1)).^2));
R2 = corrcoef(PRE,True);
R2 = R2(1,2)^ 2;
toc
time=toc;
EVALUATION=[mean(MAE) mean(MAPE) mean(RMSE) mean(IA)  R2 mean(U1_one) mean(U2_one) ];
