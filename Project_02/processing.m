clear all
close all
%% Setup
% FILENAME='3-project_time series data_students.xlsx';
% vector=xlsread(FILENAME,strcat('A1:A275'));
load('data.mat')
windowSize=15;
predictionSize=10;
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


fig10=figure(10);hold on;grid on;
set(fig10,'units','points','position',[200,550,1200,300])
% plot(vector);
for i=0:shiftSize
    plot((trainStart+i:trainEnd-shiftSize+i),trainData(windowSize+1+i,:)','lineWidth',predictionSize+1-i);
    plot((testStart:testEnd),testData)
    
end