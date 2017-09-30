clear all
close all
%% Setup
% FILENAME='3-project_time series data_students.xlsx';
% vector=xlsread(FILENAME,strcat('A1:A275'));
load('data.mat')
windowSize=10;
predictionSize=1;
testSize=30;
totalSize=windowSize+predictionSize;
vecLen=length(vector);
shiftSize=predictionSize-1;
totalShift=totalSize-1;
% figure(1);grid on;plot(vector);
%% Organize Data
trainStart=windowSize+1;
trainEnd=vecLen-testSize;
testStart= trainEnd+1;
testEnd= vecLen;

for i=trainStart+shiftSize:trainEnd
    trainData(:,i-windowSize-shiftSize)=vector(i-totalShift:i);
end

testData=vector(testStart:testEnd)';

% fig10=figure(10);hold on;grid on;
% set(fig10,'units','points','position',[200,550,1200,300])
% % plot(vector);
% for i=0:shiftSize
%     plot((trainStart+i:trainEnd-shiftSize+i),trainData(windowSize+1+i,:)','lineWidth',predictionSize+1-i);
%     plot((testStart:testEnd),testData)
%     
% end

%% Train all
m=1;
% Train.Algorithm='trainbr';
Train.Algorithm='trainlm';
Train.Lrate=0.0001;
Train.Hneurons=10;

% prep training input
order = randperm(length(trainData));
randtrainData = trainData(:,order);

% setup net
net{m} = fitnet(Train.Hneurons,Train.Algorithm);
net{m} = configure(net{m},randtrainData(1:windowSize,:),randtrainData(windowSize+1:end,:));
net{m}.trainParam.lr = Train.Lrate;
net{m}.trainParam.epochs=10;% Number of Iterations
net{m}.divideParam.trainRatio = 0.95;
net{m}.divideParam.valRatio = 0.05;
net{m}.divideParam.testRatio = 0.0;
net{m}.trainParam.max_fail=20;


% training algorithm
net{m} = train(net{m},randtrainData(1:windowSize,:),randtrainData(windowSize+1:end,:));

%% testing
train_result = net{m}(trainData(1:windowSize,:));

test_vec=trainData(trainEnd-windowSize+1:trainEnd);

for i=1:predictionSize:testSize
    test_result(1,(i:i-1+predictionSize)) = net{m}(test_vec(i:i+windowSize-1)');
    test_vec(windowSize+i:windowSize+i-1+predictionSize) = test_result(1,(i:i-1+predictionSize));
end

%% Plotting

fig1=figure(1)
% fig1.Renderer='Painters';
set(fig1,'units','points','position',[200,450,1200,300])
hold on;grid on;
% plot actual
plot(vector,'Color',[.5 .5 .5],'lineWidth',2)
plot((testStart:testEnd),testData,'Color',[.1 .1 .1],'lineWidth',2)
% plot results
% for o=0:predictionSize-1
%     plot((trainStart+o:trainEnd+o),train_result(o+1,:),'b','lineWidth',2)
% end
for i=0:shiftSize
    plot((trainStart+i:trainEnd-shiftSize+i),train_result(i+1,:)','lineWidth',1);
end
plot((testStart:testEnd),test_result,'r','lineWidth',2)

%% MSE Training Calculation
errTrain = immse(trainData(windowSize+1,:),train_result(1,:))
errTest = immse(testData,test_result)







