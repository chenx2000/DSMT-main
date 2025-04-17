%% plot color pics
clear; clc;
load(['simulation_results\results\','truth','.mat']);

load(['simulation_results\results\','hdnet','.mat']);
pred_block_hdnet = pred;

load(['simulation_results\results\','tsanet','.mat']);
pred_block_tsanet = pred;

load(['simulation_results\results\','mst','.mat']);
pred_block_mst = pred;

load(['simulation_results\results\','cst_l_plus','.mat']);
pred_block_cst_l_plus = pred;

load(['simulation_results\results\','dsmt','.mat']);
pred_block_dsmt = pred;

lam28 = [453.5 457.5 462.0 466.0 471.5 476.5 481.5 487.0 492.5 498.0 504.0 510.0...
    516.0 522.5 529.5 536.5 544.0 551.5 558.5 567.5 575.5 584.5 594.5 604.0...
    614.5 625.0 636.5 648.0];

truth(find(truth>0.7))=0.7;
pred_block_hdnet(find(pred_block_hdnet>0.7))=0.7;
pred_block_tsanet(find(pred_block_tsanet>0.7))=0.7;
pred_block_mst(find(pred_block_mst>0.7))=0.7;
pred_block_cst_l_plus(find(pred_block_cst_l_plus>0.7))=0.7;
pred_block_dsmt(find(pred_block_dsmt>0.7))=0.7;

f = 2;

%% plot spectrum
figure(123);
[yx, rect2crop]=imcrop(sum(squeeze(truth(f, :, :, :)), 3));
rect2crop=round(rect2crop)
close(123);

figure; 

spec_mean_truth = mean(mean(squeeze(truth(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_hdnet = mean(mean(squeeze(pred_block_hdnet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_tsanet = mean(mean(squeeze(pred_block_tsanet(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_mst = mean(mean(squeeze(pred_block_mst(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_cst_l_plus = mean(mean(squeeze(pred_block_cst_l_plus(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);
spec_mean_dsmt = mean(mean(squeeze(pred_block_dsmt(f,rect2crop(2):rect2crop(2)+rect2crop(4) , rect2crop(1):rect2crop(1)+rect2crop(3),:)),1),2);

spec_mean_truth = spec_mean_truth./max(spec_mean_truth);
spec_mean_hdnet = spec_mean_hdnet./max(spec_mean_hdnet);
spec_mean_tsanet = spec_mean_tsanet./max(spec_mean_tsanet);
spec_mean_mst = spec_mean_mst./max(spec_mean_mst);
spec_mean_cst_l_plus = spec_mean_cst_l_plus./max(spec_mean_cst_l_plus);
spec_mean_dsmt = spec_mean_dsmt./max(spec_mean_dsmt);

corr_hdnet = roundn(corr(spec_mean_truth(:),spec_mean_hdnet(:)),-4);
corr_tsanet = roundn(corr(spec_mean_truth(:),spec_mean_tsanet(:)),-4);
corr_mst = roundn(corr(spec_mean_truth(:),spec_mean_mst(:)),-4);
corr_cst_l_plus = roundn(corr(spec_mean_truth(:),spec_mean_cst_l_plus(:)),-4);
corr_dsmt = roundn(corr(spec_mean_truth(:),spec_mean_dsmt(:)),-4);

X = lam28;

Y(1,:) = spec_mean_truth(:); 
Y(2,:) = spec_mean_hdnet(:); Corr(1)=corr_hdnet;
Y(3,:) = spec_mean_tsanet(:); Corr(2)=corr_tsanet;
Y(4,:) = spec_mean_mst(:); Corr(3)=corr_mst;
Y(5,:) = spec_mean_cst_l_plus(:); Corr(4)=corr_cst_l_plus;
Y(6,:) = spec_mean_dsmt(:); Corr(5)=corr_dsmt;

createfigure(X,Y,Corr)

