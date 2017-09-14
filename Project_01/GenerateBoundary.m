%% Boundary Function 
% Function takes about 30s to finish each
function [trajectory]=GenerateBoundary(net)
% global check
xtest=(-15:1:25);
ytest=(-2:.1:15);
temp=0;

check=0;
for n=1:length(xtest)
    m=0;
    while check<.5
        m=m+1;
        check=net([xtest(n);ytest(m)]);
        if check>=.5
            trajectory(1:2,n)=[xtest(n);ytest(m)];
            temp=(ytest(m)-1:.01:ytest(m)+3);
        end
    end
    check=0;
    ytest=[];
    ytest=temp;
end


end