%% Cluster Function
function [C1Train,C2Train,C1Test,C2Test]=GenerateClusters(d)
% Variables from Project Description
r=10;
w=6;

% Seeds
seed1=1;    seed5=5;
seed2=2;    seed6=6;
seed3=3;    seed7=7;
seed4=4;    seed8=8;

% Cluster 1 Training (Rho-Magnitude Theta-angle[rad])
rng(seed1,'twister');
Cluster1RhoTrain = (r-w/2)+w*rand(1,1000);
rng(seed2,'twister');
Cluster1ThetaTrain = pi*rand(1,1000);

% Cluster 2 Training (Rho-Magnitude Theta-angle[rad])
rng(seed3,'twister');
Cluster2RhoTrain = (r-w/2)+w*rand(1,1000);
rng(seed4,'twister');
Cluster2ThetaTrain = -pi*rand(1,1000);

% Cluster 1 Testing (Rho-Magnitude Theta-angle[rad])
rng(seed5,'twister');
Cluster1RhoTest = (r-w/2)+w*rand(1,1000);
rng(seed6,'twister');
Cluster1ThetaTest = pi*rand(1,1000);

% Cluster 2 Testing (Rho-Magnitude Theta-angle[rad])
rng(seed7,'twister');
Cluster2RhoTest = (r-w/2)+w*rand(1,1000);
rng(seed8,'twister');
Cluster2ThetaTest = -pi*rand(1,1000);

% Convert to Carteasian Coordinate system
[Cluster1XTrain, Cluster1YTrain] = pol2cart(Cluster1ThetaTrain,Cluster1RhoTrain);
[Cluster2XTrain, Cluster2YTrain] = pol2cart(Cluster2ThetaTrain,Cluster2RhoTrain);
[Cluster1XTest, Cluster1YTest] = pol2cart(Cluster1ThetaTest,Cluster1RhoTest);
[Cluster2XTest, Cluster2YTest] = pol2cart(Cluster2ThetaTest,Cluster2RhoTest);

% Shift Cluster2
Cluster2XTrain=Cluster2XTrain+r;
Cluster2YTrain=Cluster2YTrain-d;
Cluster2XTest=Cluster2XTest+r;
Cluster2YTest=Cluster2YTest-d;

C1Train=[Cluster1XTrain;Cluster1YTrain];
C2Train=[Cluster2XTrain;Cluster2YTrain];

C1Test=[Cluster1XTest;Cluster1YTest];
C2Test=[Cluster2XTest;Cluster2YTest];

end