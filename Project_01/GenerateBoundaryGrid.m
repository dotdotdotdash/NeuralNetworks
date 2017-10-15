%% Boundary Function
% Function takes about 30s to finish each
function [grid,X,Y]=GenerateBoundaryGrid(net)
xtest=(-15:1:25);
ytest=(15:-1:-15);
grid=zeros(length(ytest),length(xtest));
[X,Y]=meshgrid(xtest,ytest);

for n=1:length(xtest)
    for m=1:length(ytest)
       grid(m,n)=net([xtest(n);ytest(m)]);
    end
end
   
