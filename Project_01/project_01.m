%% Project 1 Clark Lakshminarayanan Sonawani
clear all
close all

%% Generate Clusters

% Training and Testing clusters for d = 2
[Train1D2,Train2D2,Test1D2,Test2D2]=GenerateClusters(2);
% Training and Testing clusters for d = -4
[Train1Dn4,Train2Dn4,Test1Dn4,Test2Dn4]=GenerateClusters(-4);
% Training and Testing clusters for d = -8
[Train1Dn8,Train2Dn8,Test1Dn8,Test2Dn8]=GenerateClusters(-8);

% % Organize Data in Clusters
TrainData={Train1D2,Train2D2;
           Train1Dn4,Train2Dn4;
           Train1Dn8,Train2Dn8};

TestData={Test1D2,Test2D2;
          Test1Dn4,Test2Dn4;
          Test1Dn8,Test2Dn8};



%% Plot Clusters
fig1=figure(1)
fig1.Renderer='Painters';
set(fig1,'units','points','position',[200,200,700,600])
hold on
grid on
% scatter(Train1D2(1,:),Train1D2(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2D2(1,:),Train2D2(2,:),20,[.6 .6 .8],'filled')
scatter(Test1D2(1,:),Test1D2(2,:),20,'r','filled')
scatter(Test2D2(1,:),Test2D2(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')


fig2=figure(2)
fig2.Renderer='Painters';
set(fig2,'units','points','position',[200,200,700,600])
hold on
grid on
% scatter(Train1Dn4(1,:),Train1Dn4(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2Dn4(1,:),Train2Dn4(2,:),20,[.6 .6 .8],'filled')
scatter(Test1Dn4(1,:),Test1Dn4(2,:),20,'r','filled')
scatter(Test2Dn4(1,:),Test2Dn4(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')


fig3=figure(3)
fig3.Renderer='Painters';
set(fig3,'units','points','position',[200,200,700,600])
hold on
grid on
% scatter(Train1Dn8(1,:),Train1Dn8(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2Dn8(1,:),Train2Dn8(2,:),20,[.6 .6 .8],'filled')
scatter(Test1Dn8(1,:),Test1Dn8(2,:),20,'r','filled')
scatter(Test2Dn8(1,:),Test2Dn8(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')



%% Train all

Test.Algorithms={'traingd','traingdm','trainlm'};
Test.d=[2,-4,-8];
Test.Lrate=0.6;
Test.Nneurons=15;

for n=1:length(Test.Algorithms)
    for m=1:length(Test.d)
        rng(10);
        % prep training input
        y1 = ones(1,1000);
        y2 = zeros(1,1000);
        train_set = [TrainData{m,1},TrainData{m,2}];
        order = randperm(2000);
        train_set = train_set(:,order);
        target = [y1,y2];
        target = target(order);

        % setup net
        net{m,n} = feedforwardnet(Test.Nneurons,Test.Algorithms{n});
        net{m,n} = configure(net{m,n},train_set,target);
        net{m,n}.trainParam.lr = Test.Lrate;
        net{m,n}.trainParam.epochs=20000;% Number of Iterations
        net{m,n}.divideParam.trainRatio = 0.75;
        net{m,n}.divideParam.valRatio = 0.25;
        net{m,n}.trainParam.max_fail=2000;
        if m==2
            net{m,n}.trainParam.mc = 0.1;% momentum rate
        end
        
        % training algorithm
        net{m,n} = train(net{m,n},train_set,target);
        classifier{m,n} = GenerateBoundary(net{m,n});
        train_op = net{m,n}(train_set);

        % testing algorithm
        order = randperm(1000);
        testing_set = [TestData{m,1},TestData{m,2}];
        testing_set = testing_set(:,order);
        test_op = net{m,n}(testing_set);
        
        
        figure(m)
        hold on;grid on;
        plot(classifier{m,n}(1,:),classifier{m,n}(2,:),'lineWidth',2)
        xlim([-15 25]);
        ylim([-15 15]);
        
    end
end


%% Diplay
% for m=1:length(Test.d)
% 
%     figure(m)
%     hold on;grid on;
%     plot(classifier{m,1}(1,:),classifier{m,1}(2,:))
%     plot(classifier{m,2}(1,:),classifier{m,2}(2,:))
%     plot(classifier{m,3}(1,:),classifier{m,3}(2,:))
% 
% end


figure(1)
legend('Cluster 1 (d=2)','Cluster 2 (d=2)','Boundary: Back Propogation','Boundary: Back Propogation with Momentum','Boundary: Levenberg Marquardt')
print('-painters','-deps','D2_Lp6')
figure(2)
legend('Cluster 1 (d=2)','Cluster 2 (d=2)','Boundary: Back Propogation','Boundary: Back Propogation with Momentum','Boundary: Levenberg Marquardt')
print('-painters','-deps','Dn4_Lp6')
figure(3)
legend('Cluster 1 (d=2)','Cluster 2 (d=2)','Boundary: Back Propogation','Boundary: Back Propogation with Momentum','Boundary: Levenberg Marquardt')
print('-painters','-deps','Dn8_Lp6')












