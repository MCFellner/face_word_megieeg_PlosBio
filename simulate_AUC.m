% simulate auc with peaks of different width


% wavelet: peaks should be gaussian smoothed 
% generate gaussians with different m/sd


for sd_ratio=5
figure
        
all_offset=[-6,-4,-2,0,2,4,6]
    for off=1:numel(all_offset)
        offset=all_offset(off);
        
        x=-20:0.1:20;
Y1 = normpdf(x,0,1);
Y2 = normpdf(x,offset,1*sd_ratio);

scaling=max(Y1)/max(Y2);
Y2=Y2.*scaling;

auc1=cumsum(Y1);
auc1=auc1./auc1(end);
auc2=cumsum(Y2);
auc2=auc2./auc2(end);

subplot(numel(all_offset),2,(off.*2)-1)
hold on
plot(x,Y1)
plot(x,Y2)
subplot(numel(all_offset),2,off.*2)
hold on
plot(x,auc1)
plot(x,auc2)
title(strcat('sd ratio=1:', num2str(sd_ratio),'  M-diff=',num2str(offset)))

end
end

