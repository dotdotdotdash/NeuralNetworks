%% Boundary Function
% Function takes about 30s to finish each
function [trajectory]=GenerateBoundary(net)
xtest=(-15:1:25);
ytest=(-15:.1:15);
temp=0;
flag=0;

check=0;
for n=1:length(xtest)
    m=0;
    while check<.5&&flag==0;
        m=m+1;
        if m<= length(ytest)
            check=net([xtest(n);ytest(m)]);
            if check>=.5
                trajectory(1:2,n)=[xtest(n);ytest(m)];
                temp=(ytest(m)-1:.1:ytest(m)+5);
            end
        else
            flag=1;
        end
    end
    check=0;
    ytest=[];
    ytest=temp;
    flag=0;
end
   
trajectory(1:2,end+1)=[26;1000];