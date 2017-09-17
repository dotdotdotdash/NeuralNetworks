close all;
clear all;
clc;

rng(10);

% [c1train,c2train,c1test,c2test] = GenerateClusters(2);
% BackPropagation(c1train,c2train,c1test,c2test);
% [c1train,c2train,c1test,c2test] = GenerateClusters(-4);
% BackPropagation(c1train,c2train,c1test,c2test);
[c1train,c2train,c1test,c2test] = GenerateClusters(2);
% BackPropagation(c1train,c2train,c1test,c2test);

% prep training input
    y1 = ones(1,1000);
    y2 = zeros(1,1000);
    train_set = [c1train,c2train];
    order = randperm(2000);
    train_set = train_set(:,order);
    target = [y1,y2];
    target = target(order);

    % training algorithm
    net = feedforwardnet(3,'trainlm');
    net = configure(net,train_set,target);
    net.trainParam.lr = 0.5;
    net.trainParam.epochs = 16000;
    net.divideParam.trainRatio = 0.75;
    net.divideParam.valRatio = 0.25;
    net = train(net,train_set,target);
    classifier = GenerateBoundary(net);
    train_op = net(train_set);

    % testing algorithm
    order = randperm(1000);
    testing_set = [c1test,c2test];
    testing_set = testing_set(:,order);
    label = [ones(1,500),zeros(1,500)];
    label = label(order);
    test_op = net(testing_set);
    
        figure
    hold on;
    scatter(c1train(1,:),c1train(2,:),5,'r','filled');
    scatter(c2train(1,:),c2train(2,:),5,'g','filled');
    plot(classifier(1,:),classifier(2,:),'b');

    figure
    plotconfusion(target,train_op);

    figure
    hold on;
    scatter(testing_set(1,:),testing_set(2,:),5,'r','filled');
    plot(classifier(1,:),classifier(2,:),'b');

    figure
    plotconfusion(label,test_op);

% function BackPropagation(c1train,c2train,c1test,c2test)
% 
%     % prep training input
%     y1 = ones(1,1000);
%     y2 = zeros(1,1000);
%     train_set = [c1train,c2train];
%     order = randperm(2000);
%     train_set = train_set(:,order);
%     target = [y1,y2];
%     target = target(order);
% 
%     % training algorithm
%     net = feedforwardnet(3,'trainlm');
%     net = configure(net,train_set,target);
%     net.trainParam.lr = 0.5;
%     net.trainParam.epochs = 16000;
%     net.divideParam.trainRatio = 0.75;
%     net.divideParam.valRatio = 0.25;
%     net = train(net,train_set,target);
%     classifier = GenerateBoundary(net);
%     train_op = net(train_set);
% 
%     % testing algorithm
%     order = randperm(1000);
%     testing_set = [c1test,c2test];
%     testing_set = testing_set(:,order);
%     label = [ones(1,500),zeros(1,500)];
%     label = label(order);
%     test_op = net(testing_set);
%     
%     Displayfunction(c1train,c2train,classifier,testing_set,label,target,train_op,test_op);
%     
% end
% 
% function Displayfunction(c1train,c2train,classifier,testing_set,label,target,train_op,test_op)
% 
%     % display methods
%     figure
%     hold on;
%     scatter(c1train(1,:),c1train(2,:),5,'r','filled');
%     scatter(c2train(1,:),c2train(2,:),5,'g','filled');
%     plot(classifier(1,:),classifier(2,:),'b');
% 
%     figure
%     plotconfusion(target,train_op);
% 
%     figure
%     hold on;
%     scatter(testing_set(1,:),testing_set(2,:),5,'r','filled');
%     plot(classifier(1,:),classifier(2,:),'b');
% 
%     figure
%     plotconfusion(label,test_op);
%     
% end