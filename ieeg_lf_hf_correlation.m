project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%%

path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(project_path,'ieeg_data','freq','stats_timecourse');
mkdir(path_out);

% all patients with word and face session
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
all_frq        = [2 5;50 90];
all_time     = [1 1.5;0.3 1.5];
all_effect={'theta','gamma'};
all_freqs= {'lf','hf'};


alpha_level=0.05;
min_t=10;

for roi=1:numel(all_effect)
effect=all_effect{roi};
frq        = all_frq(roi,:);
time     = all_time(roi,:);
freq_def=all_freqs{roi};
for n=1:numel(pat)

load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_faces')))
data1=freq;
load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_words')))
data2=freq;
clear freq

% get toi and foi 
f1=nearest(data1.freq,frq(1));
f2=nearest(data1.freq,frq(2));
t1=nearest(data1.time,time(1));
t2=nearest(data1.time,time(2));
% define trials
hitind1=find(data1.trialinfo(:,4)==1);
hitind2=find(data2.trialinfo(:,4)==1);
missind1=find(data1.trialinfo(:,4)==0);
missind2=find(data2.trialinfo(:,4)==0);

trl_info(:,1)=[ones(size(data1.trialinfo,1),1);ones(size(data2.trialinfo,1),1).*2];
trl_info(:,2)=[data1.trialinfo(:,4);data2.trialinfo(:,4)];

 freq1=squeeze(nanmean(nanmean(data1.powspctrm(hitind1,:,f1:f2,t1:t2),3),4)); 
 freq2=squeeze(nanmean(nanmean(data1.powspctrm(missind1,:,f1:f2,t1:t2),3),4));   
 freq3=squeeze(nanmean(nanmean(data2.powspctrm(hitind2,:,f1:f2,t1:t2),3),4)); 
 freq4=squeeze(nanmean(nanmean(data2.powspctrm(missind2,:,f1:f2,t1:t2),3),4));  

 freq_time=[squeeze(nanmean(data1.powspctrm(:,:,f1:f2,:),3));squeeze(nanmean(data2.powspctrm(:,:,f1:f2,:),3))]; 

 
hit1=sum(~isnan(freq1));
hit2=sum(~isnan(freq2));
miss1=sum(~isnan(freq3));
miss2=sum(~isnan(freq4));
 
 elec_trial_num_ok=(hit1>=min_t & hit2>=min_t & miss1>=min_t &miss2>=min_t);

ok_elec=find(elec_trial_num_ok==1);

 freq1=freq1(:,ok_elec);
 freq2=freq2(:,ok_elec);
 freq3=freq3(:,ok_elec);
 freq4=freq4(:,ok_elec);
  
 freq_time=freq_time(:,ok_elec,:);

 freq=[freq1;freq2;freq3;freq4];

mem=[repmat({'hit'},size(hitind1));repmat({'miss'},size(missind1));repmat({'hit'},size(hitind2));repmat({'miss'},size(missind2))];
cond=[repmat({'word'},size([hitind1;missind1]));repmat({'face'},size([hitind2;missind2]))];
  
    for e=1:size(freq1,2)

        [p_tmp,tbl,stats] = anovan(freq(:,e),{mem cond},'model','interaction','varnames',{'mem','cond'},'display','off');               
        h(e,:)=double(p_tmp'<alpha_level);  
        p(e,:)=p_tmp';
       % 1:mem, 2:cond, 3:interaction
           
    end
     direction(:,1)=sign((((nanmean(freq1)+nanmean(freq3)).*0.5)-((nanmean(freq2)+nanmean(freq4)).*0.5)));
     direction(:,2)=sign((((nanmean(freq1)+nanmean(freq2)).*0.5)-((nanmean(freq3)+nanmean(freq4)).*0.5)));
     direction(:,3)=sign(((nanmean(freq1)-nanmean(freq2))-(nanmean(freq3)-nanmean(freq4))));

     
allelec_stat{n,roi}.h=h;
allelec_stat{n,roi}.p=p;
allelec_stat{n,roi}.direction=direction;
allelec_stat{n,roi}.elecs=data1.label(ok_elec);
allelec_stat{n,roi}.freq_time=freq_time;
allelec_stat{n,roi}.trl_info=trl_info;
allelec_stat{n,roi}.time=data1.time;
allelec_stat{n,roi}.effects={'mem','cond','interaction'};

clear direction h p_tmp tbl stats hitind1 hitind2 missind1 missind2 trl_info freq_time
     
clear mem  cond freq1 freq2 freq3 freq4 freq data1 data2 f1 f2 t1 t2 
end
end
save(fullfile(path_out,'all_elecstat'),'allelec_stat','all_frq', 'all_effect', 'all_time','pat')


%%
path_figs=fullfile(project_path,'figures');
path_in=fullfile(project_path,'ieeg_data','freq','stats_timecourse');
load(fullfile(path_in,'all_elecstat'))

elec_defs={'theta_elec','gamma_elec'};
fig_names={'fig5f','fig5c'};
for d=1:numel(elec_defs)
    elec_def=elec_defs{d};
    fig_name=fig_names{d};;
switch elec_def
    case 'theta_elec'
%correlate theta-gamma in theta electrodes (sig SME)
roi=1;
foi1=1;
foi2=2;
eff=1;
direction=-1;
plottitle='corr theta gamma in theta SME electrodes'
    case 'gamma_elec'
% correlate theta-gamma in gamma electrodes (sig SME)
roi=2;
foi1=1;
foi2=2;
eff=1;
direction=1;
plottitle='corr theta gamma in gamma SME electrodes'
end


% select nonnan timewindows
sel_toiind=51:101;
toi=allelec_stat{1,1}.time(sel_toiind);
n_rand=1000;
p_cluster=0.05;
% random dist
% get electrodes & freq time course
for r=1:n_rand
    display(strcat('random ', num2str(r)))
for p=1:numel(pat)
sel_elec=find((allelec_stat{p,roi}.h(:,eff).*allelec_stat{p,roi}.direction(:,eff))==direction);
corr_tmp=[];
corr_rand=[];
for e=1:numel(sel_elec)

sel_freq1=squeeze(allelec_stat{p,foi1}.freq_time(:,sel_elec(e),sel_toiind));
sel_freq2=squeeze(allelec_stat{p,foi2}.freq_time(:,sel_elec(e),sel_toiind));
% check for NaNs
nonan=find(~isnan(nanmean(sel_freq1,2)));

%corr_tmp(e,:,:)=corr(sel_freq1(nonan,:),sel_freq2(nonan,:));

rand_ind=randperm(size(nonan,1));

corr_rand(e,:,:)=corr(sel_freq1(nonan,:),sel_freq2(nonan(rand_ind),:));

end
corr_randall{p}.corr=corr_rand;
end


% plot mean correlation
corr_vec=[];
for p=1:numel(pat)
corr_vec=[corr_vec;corr_randall{p}.corr];
end

%fisher z
corr_vec=  0.5.*log(((ones(size(corr_vec))+corr_vec)./(ones(size(corr_vec))-corr_vec)));

[h,~,~,stat]=ttest((corr_vec));

 rand_mask_neg=squeeze(h.*(stat.tstat<0));
    rep=find(isnan(rand_mask_neg));      
    rand_mask_neg(rep)=0;         
    [rand_L_neg,rand_num_neg] = bwlabel(rand_mask_neg);

        for neg=1:rand_num_neg
            m=(rand_L_neg==neg);
            rand_negt(neg)=sum(sum(squeeze(stat.tstat).*m));
        end
         if isempty(neg)
            rand_negt=0;
         end

    rand_mask_pos=squeeze(h.*(stat.tstat>0));
    rep=find(isnan(rand_mask_pos));
    rand_mask_pos(rep)=0;     
        [rand_L_pos,rand_num_pos] = bwlabel(rand_mask_pos);    
        for pos=1:rand_num_pos
            m=(rand_L_pos==pos);
            rand_post(pos)=sum(sum(squeeze(stat.tstat).*m));
        end
        if isempty(pos)
            rand_post=0;
        end

 pos_tsum{r}=sort(rand_post,'descend');
 neg_tsum{r}=sort(rand_negt,'ascend');
  
end

max_pos=max(cellfun(@numel,pos_tsum));
max_neg=max(cellfun(@numel,neg_tsum));

for x=1:n_rand
   pos_tsum{x}= [pos_tsum{x},zeros(1,max_pos-numel(pos_tsum{x}))];
   neg_tsum{x}= [neg_tsum{x},zeros(1,max_neg-numel(neg_tsum{x}))];
end

pos_dist=reshape([pos_tsum{:}],max_pos,n_rand);
pos_dist=sort(pos_dist,2,'descend');

neg_dist=reshape([neg_tsum{:}],max_neg,n_rand);
neg_dist=sort(neg_dist,2,'ascend');

clear rand_L_neg rand_L_pos rand_num_neg rand_num_pos ci ans neg_tsum pos_tsum
clear rand_data rand_def_vec rand_inhibtion rand_mask_neg rand_mask_pos
clear rand_negt rand_post rand_rehearsal rep rand_vec pos neg 



% data
for p=1:numel(pat)
sel_elec=find((allelec_stat{p,roi}.h(:,eff).*allelec_stat{p,roi}.direction(:,eff))==direction);
corr_tmp=[];
corr_rand=[];
for e=1:numel(sel_elec)

sel_freq1=squeeze(allelec_stat{p,foi1}.freq_time(:,sel_elec(e),sel_toiind));
sel_freq2=squeeze(allelec_stat{p,foi2}.freq_time(:,sel_elec(e),sel_toiind));
% check for NaNs
nonan=find(~isnan(nanmean(sel_freq1,2)));

%corr_tmp(e,:,:)=corr(sel_freq1(nonan,:),sel_freq2(nonan,:));

rand_ind=randperm(size(nonan,1));

corr_mat(e,:,:)=corr(sel_freq1(nonan,:),sel_freq2(nonan,:));

end
corr_all{p}.corr=corr_mat;
end


% plot mean correlation
corr_vec=[];
for p=1:numel(pat)
corr_vec=[corr_vec;corr_all{p}.corr];
end

%fisher z
corr_vec=  0.5.*log(((ones(size(corr_vec))+corr_vec)./(ones(size(corr_vec))-corr_vec)));


[h,~,~,stat]=ttest((corr_vec));
data_mask_neg=squeeze(h.*(stat.tstat<0));
rep=find(isnan(data_mask_neg));      
data_mask_neg(rep)=0;         
[data_L_neg,data_num_neg] = bwlabel(data_mask_neg);
    
    for neg=1:data_num_neg
        m=(data_L_neg==neg);
        data_negt(neg)=sum(sum(squeeze(stat.tstat).*m));
    end
     if isempty(neg)
        data_negt=0;
     end
    [data_negt,ind_negt]=sort(data_negt,'ascend');
       
data_mask_pos=squeeze(h.*(stat.tstat>0));
rep=find(isnan(data_mask_pos));
data_mask_pos(rep)=0;     
    [data_L_pos,data_num_pos] = bwlabel(data_mask_pos);    
    for pos=1:data_num_pos
        m=(data_L_pos==pos);
        data_post(pos)=sum(sum(squeeze(stat.tstat).*m));
    end
    if isempty(pos)
        data_post=0;
    end
    [data_post,ind_post]=sort(data_post,'descend');
    % keep tstat for later plotting   
    data_stat=stat;


    
 % significance?
    % check pos clusters
     ind=1; 
     p_threshold=p_cluster;
     sig_mask_pos=zeros(size(data_mask_pos));
     pos_cluster_p=[];
     pos_cluster(ind)=nearest(pos_dist(ind,:),data_post(ind))./n_rand;
     while nearest(pos_dist(ind,:),data_post(ind))<=round(p_threshold.*n_rand)
        p_threshold=p_threshold/2;
        sig_mask_pos=sig_mask_pos+(data_L_pos==ind_post(ind));
        pos_cluster(ind)=nearest(pos_dist(ind,:),data_post(ind))./n_rand;
        ind=ind+1;
       if ind>numel(data_post)|ind> max_pos
        break
       end

     end

     
% check neg clusters
     ind=1; 
     p_threshold=p_cluster;
     sig_mask_neg=zeros(size(data_mask_neg));
     neg_cluster_p=[];
     neg_cluster(ind)=nearest(neg_dist(ind,:),data_negt(ind))./n_rand;
     while nearest(neg_dist(ind,:),data_negt(ind))<=round(p_threshold.*n_rand) 
        p_threshold=p_threshold/2;
        sig_mask_neg=sig_mask_neg+(data_L_neg==ind_negt(ind));
        neg_cluster(ind)=nearest(neg_dist(ind,:),data_negt(ind))./n_rand;
        ind=ind+1;
        
       if ind>numel(data_negt)|ind> max_neg
      break
       end
       
     end     
     
     
clear ind p_threshold      
% plot results
mask_alpha=sig_mask_pos+sig_mask_neg;
        ind_z=find(mask_alpha==0);
        mask_alpha(ind_z)=0.5;
   

figure
H=imagesc(toi,toi,squeeze((stat.tstat)),[-4 4])
set(H,'AlphaData',mask_alpha)

set(gca,'YDir','normal')
foi1_label=all_effect{foi1};
foi2_label=all_effect{foi2};
ylabel(foi1_label)
xlabel(foi2_label)
title(strcat(plottitle,' neg cluster:p=',num2str(neg_cluster),' pos cluster:p=',num2str(pos_cluster)))
colorbar
colormap('jet')

savefig(fullfile(path_figs,fig_name))
close all
end
