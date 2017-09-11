clear all

%% Generate Clusters

% Training and Testing clusters for d = 2
[Train1D2,Train2D2,Test1D2,Test2D2]=GenerateClusters(2);
% Training and Testing clusters for d = -4
[Train1Dn4,Train2Dn4,Test1Dn4,Test2Dn4]=GenerateClusters(-4);
% Training and Testing clusters for d = -8
[Train1Dn8,Train2Dn8,Test1Dn8,Test2Dn8]=GenerateClusters(-8);


%% Plot Clusters
% figure(1)
% hold on
% grid on
% scatter(Train1D2(1,:),Train1D2(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2D2(1,:),Train2D2(2,:),20,[.6 .6 .8],'filled')
% scatter(Test1D2(1,:),Test1D2(2,:),20,'r','filled')
% scatter(Test2D2(1,:),Test2D2(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')
% hold off;
% figure(2)
% hold on
% grid on
% scatter(Train1Dn4(1,:),Train1Dn4(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2Dn4(1,:),Train2Dn4(2,:),20,[.6 .6 .8],'filled')
% scatter(Test1Dn4(1,:),Test1Dn4(2,:),20,'r','filled')
% scatter(Test2Dn4(1,:),Test2Dn4(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')
% 
% figure(3)
% hold on
% grid on
% scatter(Train1Dn8(1,:),Train1Dn8(2,:),20,[.8 .6 .6],'filled')
% scatter(Train2Dn8(1,:),Train2Dn8(2,:),20,[.6 .6 .8],'filled')
% scatter(Test1Dn8(1,:),Test1Dn8(2,:),20,'r','filled')
% scatter(Test2Dn8(1,:),Test2Dn8(2,:),20,'b','filled')
% legend('Cluster 1 Training','Cluster 2 Training','Cluster 1 Testing','Cluster 2 Testing')

%% train  Clusters back propogation by momentum

X1=[Train1D2 Train2D2];
X2=[Train1Dn4 Train2Dn4];
X3=[Train2Dn8 Train2Dn8];
y = [zeros(1000,1);ones(1000,1)]';
A=[X1; y]
B=[X2; y]
C=[X3; y]
rand=randperm(2000,2000);
A=A(:,rand); %randmized inputs and targets
B=B(:,rand);
C=C(:,rand);

net = feedforwardnet([35],'trainrp');%%resilient backpropogation with momentum
% net_1 = feedforwardnet([10],'trainrp');
% net_1 = feedforwardnet([10],'trainrp');

net = configure(net,A(1:2,:),A(3,:) );
net.trainParam.epochs=4000;% Number of Iterations
net.trainParam.lr = 0.001;% learning rate
net.trainParam.mc = 0.9;% momentum rate
net.trainParam.max_fail=2000;
net = init(net);
view(net);


net_1=train(net,A(1:2,:),A(3,:));
net_2=train(net,B(1:2,:),B(3,:));
net_3=train(net,C(1:2,:),C(3,:));


X_test1=[Test1D2 Test2D2]
output_1= net_1(X_test1);
for i=1:1000
    if(output_1(i)>=0.5)
        output_1(i)=1;
    else
         output_1(i)=2;
    end
end


X_test2=[Test1Dn4 Test2Dn4]
output_2= net_2(X_test2);
for i=1:1000
    if(output_2(i)>=0.5)
        output_2(i)=1;
    else
         output_2(i)=2;
    end
end

X_test3=[Test1Dn8 Test2Dn8]
output_3= net_3(X_test3);
for i=1:1000
    if(output_3(i)>=0.5)
        output_3(i)=1;
    else
         output_3(i)=2;
    end
end




figure(4);
gscatter(X_test1(1,:),X_test1(2,:),output_1,'rb','o+');
figure(5);
gscatter(X_test2(1,:),X_test2(2,:),output_2,'rb','o+');
figure(6);
gscatter(X_test3(1,:),X_test3(2,:),output_3,'rb','o+');
 
legend('class 1','class 2')

