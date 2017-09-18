%% Project 1 Clark Lakshminarayanan Sonawani
clear all

%% Generate Clusters

% Training and Testing clusters for d = 2
[Train1D2,Train2D2,Test1D2,Test2D2]=GenerateClusters(2);
% Training and Testing clusters for d = -4
[Train1Dn4,Train2Dn4,Test1Dn4,Test2Dn4]=GenerateClusters(-4);
% Training and Testing clusters for d = -8
[Train1Dn8,Train2Dn8,Test1Dn8,Test2Dn8]=GenerateClusters(-8);


%% Plot Clusters
figure(1)
hold on
grid on
scatter(Train1D2(1,:),Train1D2(2,:),20,[.8 .6 .6],'filled')
scatter(Train2D2(1,:),Train2D2(2,:),20,[.6 .6 .8],'filled')
scatter(Test1D2(1,:),Test1D2(2,:),20,'r','filled')
scatter(Test2D2(1,:),Test2D2(2,:),20,'b','filled')
legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')

figure(2)
hold on
grid on
scatter(Train1Dn4(1,:),Train1Dn4(2,:),20,[.8 .6 .6],'filled')
scatter(Train2Dn4(1,:),Train2Dn4(2,:),20,[.6 .6 .8],'filled')
scatter(Test1Dn4(1,:),Test1Dn4(2,:),20,'r','filled')
scatter(Test2Dn4(1,:),Test2Dn4(2,:),20,'b','filled')
legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')

figure(3)
hold on
grid on
scatter(Train1Dn8(1,:),Train1Dn8(2,:),20,[.8 .6 .6],'filled')
scatter(Train2Dn8(1,:),Train2Dn8(2,:),20,[.6 .6 .8],'filled')
scatter(Test1Dn8(1,:),Test1Dn8(2,:),20,'r','filled')
scatter(Test2Dn8(1,:),Test2Dn8(2,:),20,'b','filled')
legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')



















