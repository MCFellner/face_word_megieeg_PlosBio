project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));



 %% redo source analysis (wavelet from 2-100Hz for slope fitting)
% 
% 
% 
% path_in=fullfile(project_path,'meg_data');
% path_sourcemodel=fullfile(project_path,'meg_data','sourcemodels');
% path_out_data=fullfile(project_path,'meg_data','source_noz');
% path_out_z=fullfile(project_path,'meg_data','source_zinfo');
% 
% 
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%         'vp31';'vp32'};
%  
% cond={'words', 'faces'};    
% % 
% 
% for c=1:numel(cond)
% for n=1:numel(sub)
% 
% 
%      load(fullfile(path_in,strcat(sub{n},'_',cond{c})));    
%      load(fullfile(path_sourcemodel,strcat(sub{n},'sourcemodel10')))
%    
% sel_trials=find(data.trialinfo(:,1)); %selects all trials
% %covariance matrix
%   cfg=[];
%         cfg.dftfilter='yes';
%         cfg.dftfilter=[50,100,150];
%         cfg.trials=sel_trials;
%         data=ft_preprocessing(cfg,data);
% cfg=[];
% cfg.covariance         = 'yes';
% cfg.covariancewindow   = 'all';
% cfg.keeptrials   = 'yes';
% 
% erp=ft_timelockanalysis(cfg,data);
% 
% % LCMV game
% 
% cfg             = [];
% cfg.grid        = sourcemodel;
% cfg.vol         = vol;
% cfg.channel     = {'MEG'};
% cfg.grad        = data.grad;
% 
% sourcemodel_lf  = ft_prepare_leadfield(cfg, data);
% 
% 
% cfg              = []; 
% cfg.method       = 'lcmv';
% cfg.grid = sourcemodel_lf;
% cfg.vol          = vol;
% cfg.lcmv.lambda       = '5%';
% cfg.lcmv.projectnoise = 'yes';
% cfg.lcmv.keepfilter   = 'yes';
% cfg.lcmv.realfilter   = 'yes';
% cfg.lcmv.fixedori   = 'yes';
% cfg.grad        = data.grad;%cfg.sensetype='eeg';
% sourceAll = ft_sourceanalysis(cfg, erp);
% 
% clear erp
% 
% %beam every single trial
% 
% %combine all trials in a matrix (chan*(time*trials))
% trials= [data.trial{1,sel_trials}];
% %combine all filters in one matrix(insidepos*chan)
% insidepos=find(sourceAll.inside);
% filters=vertcat(sourceAll.avg.filter{insidepos,:});
% 
% virtualsensors=filters*trials;
% 
% beameddata=data;
% 
% trialarray=reshape(virtualsensors,[numel(insidepos),size(data.time{1,1},2),numel(sel_trials)]);
% 
% trial=squeeze(num2cell(trialarray,[1,2]))';
% beameddata.trial=trial;
% beameddata.label=cellstr(num2str(insidepos));
% beameddata.time=data.time(sel_trials);
% beameddata.trialinfo=data.trialinfo(sel_trials,:);
% beameddata=rmfield(beameddata, 'grad');
% beameddata=rmfield(beameddata, 'sampleinfo');
% 
% clear virtualsensors trials trialarray filters trial data
% save (strcat(path_out_data,'sourceAll',sub{n},cond{c}),'sourceAll')
% 
% clear sourceAll
% clear sourcemodel sourcemodel_lf 
% 
% 
%     cfg=[];
%           cfg.method = 'wavelet';
%           cfg.width = 5;
%           cfg.output     = 'pow';
%           cfg.foi = 3:1:100;
%           cfg.toi = -0.5:0.1:1.5;
%           cfg.keeptrials = 'yes';
%           freq = ft_freqanalysis(cfg, beameddata);
%         cfg.time=[freq.time(1) freq.time(end)];
%         freq_z=z_trans_TF_seltime(cfg,freq);
%         z_trans_info.mean=freq_z.cfg.mean;
%          z_trans_info.std=freq_z.cfg.std;
% 
%          clear freq_z
%          save (strcat(path_out_z,sub{n},cond{c},'_z_info'), 'z_trans_info')
%           clear  beameddata 
% 
%          freq_all=freq;
%          clear freq
%          
% cond_sme={'hit','miss'};
% 
%     for s= 1: numel(cond_sme)   
%     
%   condition=cond_sme{s};
%     switch condition
%          case 'hit'
%             trials=find(freq_all.trialinfo(:,4)==1);
%         case 'miss'
%             trials=find(freq_all.trialinfo(:,4)==0);
%             otherwise
%     end
%       
%     freq_cond=freq_all;
%     freq_cond.dimord='chan_freq_time';
%     freq_cond.powspctrm=squeeze(nanmean(freq_all.powspctrm(trials,:,:,:),1));   
%     
%     save (strcat(path_out_data,sub{n},'beamed_TF_noz',condition,'_',cond{c}), 'freq_cond')
%     clear freq_cond 
%     end
%     clear freq_all
% end
% end
%% fit averages
% path_in=fullfile(project_path,'meg_data','source_noz');
% path_out=fullfile(project_path,'meg_data','source','source_tiltfit');

%  
%  condition={'hit_words','miss_words','hit_faces','miss_faces'};
% vp={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp28';'vp29';'vp30';...
%         'vp31';'vp32'};
%     type='robustfit';
% allfit_foi=[2,30;30,90;2,90];
% all_type={'robustfit'};
% 
% for t=1:numel(all_type)
% type=all_type{t};
% for ff=1:size(allfit_foi)
% foi=allfit_foi(ff,:);
% for n=1:numel(vp)
% for c=1:numel(condition)
% 
% load(strcat(path_in,vp{n},'beamed_TF_noz',condition{c}))
% % freq=rmfield(freq,'cfg');
% %org_freq{n,c}=freq;
% % robustfit
%          cfg.toi=[-0.5,1.5];
%          cfg.fit_type=type;
%          cfg.freq2fit= foi;
%          [freq]=sh_subtr1of(cfg,freq_cond);
%          all_freqcond{c}=freq;
% end
% save(strcat(path_out,'GA_',vp{n},type,'_fit',num2str(foi(1)),'to',num2str(foi(2)),'Hz'), 'all_freqcond','condition')
% 
%     end
% 
% end
% end
% 
% 
% 
% for n=1:numel(vp)
%    for c=1:numel(condition)
%      load(strcat(path_in,vp{n},'beamed_TF_noz',condition{c}))
%      org_pow{c}(n,:,:,:)=freq_cond.powspctrm;% add linfit to logpow
%    end
% end
% save(fullfile(path_out,'noztrans_pow_all_conditions'), 'org_pow', 'condition','-v7.3')
%% get data in source rois
%%%%%%%%%%%%% data definition
all_specific={'robustfit'};
specific=all_specific{1};
type=specific;
% 
vp={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
% 
path_in=fullfile(project_path,'meg_data','source','source_tiltfit');
path_source=fullfile(project_path,'meg_data','source');

path_fig=fullfile(project_path,'figures');


condition={'words_hit','words_miss','faces_hit','faces_miss'};
allfit_foi=[2,30;30,90;2,90];


% %%%%%%%%%%%%%meg cluster stat definition
 path_stats=fullfile(project_path,'meg_data','source','stats');

% 
% load MEG w/f rois (sourcestats...)
all_freq={'theta','alphabeta','gamma'};
freq_def=[2,5;8 20;50 90];
time_def=[1, 1.5;0.3, 1.5;0.3 1];

all_con={'main_sme'};
% 

for con=1;

sel_con=all_con{con};
for f=1:numel(all_freq);

sel_freq=all_freq{f};
foi_stat=freq_def(f,:);
toi=time_def(f,:);

  freqstr=strcat(num2str(foi_stat(1)),'to',num2str(foi_stat(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');
filename=strcat('sourcestats_freq',sel_con,freqstr,'time',timestr,'.mat');

load(fullfile(path_stats,filename));
        
  %use sig cluster
if isfield(stat,'posclusters')
if ~isempty(stat.posclusters)
    sig_level=[stat.posclusters(:).prob];
if any(sig_level<=0.05)
   sig_def= find(sig_level<=0.05);
    
pos_cluster_def= (stat.posclusterslabelmat<=sig_def(end)&stat.posclusterslabelmat>0); 
else 
    pos_cluster_def=zeros(size(stat.stat));
end
else
pos_cluster_def=zeros(size(stat.stat));
end
else
end

if isfield(stat,'negclusters')
if ~isempty(stat.negclusters)
    sig_level=[stat.negclusters(:).prob];
if any(sig_level<=0.05)
   sig_def= find(sig_level<=0.05);    
  neg_cluster_def= (stat.negclusterslabelmat<=sig_def(end)&stat.negclusterslabelmat>0); 
else 
   neg_cluster_def=zeros(size(stat.stat));
end
else
    neg_cluster_def=zeros(size(stat.stat));
end
else
neg_cluster_def=zeros(size(stat.stat));
end

template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));
load(fullfile(path_in,'noztrans_pow_all_conditions'))
load (fullfile(path_source,strcat(vp{1},'template_source'))) %load a template source, dummy for later stats and plotting

  % match indices of stat with pow
sourceAll.pos = template.sourcemodel.pos;
sourceAll.dim = template.sourcemodel.dim;
insidepos=find(sourceAll.inside);

pos_cluster_def=pos_cluster_def(insidepos);
neg_cluster_def=neg_cluster_def(insidepos); 


for ff=1:size(allfit_foi)
fit_foi=allfit_foi(ff,:);
for n=1:numel(vp)
      load(fullfile(path_in,strcat('GA_',vp{n},type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz')))
 for c=1:numel(condition)
  
      pow{ff,c}(n,:,:,:)=all_freqcond{c}.powspctrm;
      slope{ff,c}(n,:,:)=all_freqcond{c}.slope;
     offset{ff,c}(n,:,:)=all_freqcond{c}.offset;
fit_ind(ff,1)=nearest(all_freqcond{c}.freq,allfit_foi(ff,1));
fit_ind(ff,2)=nearest(all_freqcond{c}.freq,allfit_foi(ff,2));

 for sens=1:size(pow{c},2)
     for t=1:size(pow{c},4)
 linfit{ff,c}(n,sens,:,t)=[ones(size(all_freqcond{c}.freq));log10(all_freqcond{c}.freq)]'*[all_freqcond{c}.offset(sens,t);all_freqcond{c}.slope(sens,t)];
    end
 end
end      
 end  


% %%%%%%%%%% pos cluster
% % plot
 t1=nearest(all_freqcond{c}.time,toi(1));
 t2=nearest(all_freqcond{c}.time,toi(2));
 f1=nearest(all_freqcond{c}.freq,foi_stat(1));
f2=nearest(all_freqcond{c}.freq,foi_stat(2));
 
 freq_cond.freq=all_freqcond{c}.freq;
 
 
 if sum (pos_cluster_def>0)
for c=1:numel(condition)
 pos_mean_slope{ff,c}=squeeze(nanmean(nanmean(nanmean(slope{ff,c}(:,pos_cluster_def,t1:t2),1),2),3));
 pos_mean_offset{ff,c}=squeeze(nanmean(nanmean(nanmean(offset{ff,c}(:,pos_cluster_def,t1:t2),1),2),3));
 pos_mean_orgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(:,pos_cluster_def,:,t1:t2),1),2),4));
 pos_mean_pow{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(:,pos_cluster_def,:,t1:t2),1),2),4));
 pos_mean_linfit{ff,c}=squeeze(nanmean(nanmean(nanmean(linfit{ff,c}(:,pos_cluster_def,:,t1:t2),1),2),4));
pos_mean_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(nanmean(pow{ff,c}(:,pos_cluster_def,f1:f2,t1:t2),1),2),3),4));
 pos_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(nanmean(org_pow{c}(:,pos_cluster_def,f1:f2,t1:t2),1),2),3),4));

 pos_all_slope{ff,c}=squeeze(nanmean(nanmean(slope{ff,c}(:,pos_cluster_def,t1:t2),2),3));
 pos_all_offset{ff,c}=squeeze(nanmean(nanmean(offset{ff,c}(:,pos_cluster_def,t1:t2),2),3));
 pos_all_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(:,pos_cluster_def,:,t1:t2),2),4));
 pos_all_pow{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(:,pos_cluster_def,:,t1:t2),2),4));
 pos_all_linfit{ff,c}=squeeze(nanmean(nanmean(linfit{ff,c}(:,pos_cluster_def,:,t1:t2),2),4));
 pos_all_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(:,pos_cluster_def,f1:f2,t1:t2),2),3),4));
 pos_all_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(:,pos_cluster_def,f1:f2,t1:t2),2),3),4));

end
 end
% 
 if sum (neg_cluster_def>0)
for c=1:numel(condition)
 neg_mean_slope{ff,c}=squeeze(nanmean(nanmean(nanmean(slope{ff,c}(:,neg_cluster_def,t1:t2),1),2),3));
 neg_mean_offset{ff,c}=squeeze(nanmean(nanmean(nanmean(offset{ff,c}(:,neg_cluster_def,t1:t2),1),2),3));
 neg_mean_orgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(:,neg_cluster_def,:,t1:t2),1),2),4));
 neg_mean_pow{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(:,neg_cluster_def,:,t1:t2),1),2),4));
 neg_mean_linfit{ff,c}=squeeze(nanmean(nanmean(nanmean(linfit{ff,c}(:,neg_cluster_def,:,t1:t2),1),2),4));
 neg_mean_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(nanmean(pow{ff,c}(:,neg_cluster_def,f1:f2,t1:t2),1),2),3),4));
 neg_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(nanmean(org_pow{c}(:,neg_cluster_def,f1:f2,t1:t2),1),2),3),4));

 neg_all_slope{ff,c}=squeeze(nanmean(nanmean(slope{ff,c}(:,neg_cluster_def,t1:t2),2),3));
 neg_all_offset{ff,c}=squeeze(nanmean(nanmean(offset{ff,c}(:,neg_cluster_def,t1:t2),2),3));
 neg_all_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(:,neg_cluster_def,:,t1:t2),2),4));
 neg_all_pow{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(:,neg_cluster_def,:,t1:t2),2),4));
 neg_all_linfit{ff,c}=squeeze(nanmean(nanmean(linfit{ff,c}(:,neg_cluster_def,:,t1:t2),2),4));
 neg_all_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(:,neg_cluster_def,f1:f2,t1:t2),2),3),4));
 neg_all_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(:,neg_cluster_def,f1:f2,t1:t2),2),3),4));

end
 end
 
end

% fig pos cluster

% bargraph slopes
fig=figure
 if sum (pos_cluster_def>0)

for ff=1:3
subplot(2,8,ff)
bar([pos_mean_slope{ff,:}])
sel_all=[pos_all_slope{ff,:}];
hold on 
scatter(reshape(repmat(1:4,size(sel_all,1),1),[],1) ,reshape(sel_all,[],1),'ko');
ylim([-1.5,0])
h=gca;
fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)
h.XTickLabel=condition;
end
title(strcat('poscluster:','sourcestat_',sel_con,sel_freq, num2str(foi_stat(1)),'to', num2str(foi_stat(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))

sel_all_pos=[[pos_all_slope{1,:}],[pos_all_slope{2,:}],[pos_all_slope{3,:}]];

% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
% plot original power plus different slopes
subplot(2,8,4:5)
for c=1:numel(condition)
plot(log10(freq_cond.freq),log10(pos_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on


% plot slope lines for the different fits (only for fitted freqs)
for ff=1:3
   
plot(log10(freq_cond.freq(fit_ind(ff,1):fit_ind(ff,2))),(pos_mean_linfit{ff,c}(fit_ind(ff,1):fit_ind(ff,2))),line_def{ff},'Color',color_def(c,:))
end
end
legend(reshape(repmat(condition,4,1),[],1))

switch sel_con
    case 'main_sme'

      cond1=((pos_all_orgpow{1}+pos_all_orgpow{3}).*0.5);
      cond2=((pos_all_orgpow{2}+pos_all_orgpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot((log10(freq_cond.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot((log10(freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

xlabel('log10(freq)')
ylabel('log10(power)')

%title(strcat('ieeg pos ',sel_con,sel_freq,specific,'raw powspctrm'))


%figure
%plot res freq slope
for ff=1:3
subplot(2,8,5+ff)
hold on

for c=1:numel(condition)

[pks,locs]=findpeaks(log10(pos_mean_pow{ff,c}));
plot((freq_cond.freq),log10(pos_mean_pow{ff,c}),freq_cond.freq(locs),pks,'kx');
ind(c)=locs(1);
text(locs(1)+0.3,pks(1),strcat(num2str(round(freq_cond.freq(ind(c)))),'Hz'))

fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');
xlabel(fit_filename)

end

ylabel('log10 pow')

% for each fit add cond t test
switch sel_con
    case 'main_sme'
      cond1=((pos_all_pow{ff,1}+pos_all_pow{ff,3}).*0.5);
      cond2=((pos_all_pow{ff,2}+pos_all_pow{ff,4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);

[h,p]=ttest(cond1,cond2)
plot(freq_cond.freq(h==1),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')


end
legend(condition)

title(strcat('meg pos ',sel_con,sel_freq,specific,'residual freq'))
 
 else
 sel_all_pos=[];   

end
 
 
if sum (neg_cluster_def>0)

%%%%%%%%%%%%%% fig neg cluster

% bargraph slopes
for ff=1:3
subplot(2,8,ff+8)
bar([neg_mean_slope{ff,:}])
sel_all=[neg_all_slope{ff,:}];
hold on 
scatter(reshape(repmat(1:4,size(sel_all,1),1),[],1) ,reshape(sel_all,[],1),'ko');
ylim([-1.5,0])
h=gca;
fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)
h.XTickLabel=condition;
end
title(strcat('negcluster:','sourcestat_',sel_con,sel_freq, num2str(foi_stat(1)),'to', num2str(foi_stat(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))


sel_all_neg=[[neg_all_slope{1,:}],[neg_all_slope{2,:}],[neg_all_slope{3,:}]];

% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
% plot original power plus different slopes
subplot(2,8,12:13)
for c=1:numel(condition)
plot(log10(freq_cond.freq),log10(neg_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on
% plot slope lines for the different fits (only for fitted freqs)
for ff=1:3
   
plot(log10(freq_cond.freq(fit_ind(ff,1):fit_ind(ff,2))),(neg_mean_linfit{ff,c}(fit_ind(ff,1):fit_ind(ff,2))),line_def{ff},'Color',color_def(c,:))
end
end
xlabel('log10(freq)')
ylabel('log10(power)')

legend(reshape(repmat(condition,4,1),[],1))
switch sel_con
    case 'main_sme'

      cond1=((neg_all_orgpow{1}+neg_all_orgpow{3}).*0.5);
      cond2=((neg_all_orgpow{2}+neg_all_orgpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot((log10(freq_cond.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot((log10(freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')


title(strcat('meg neg ',sel_con,sel_freq,specific,'raw powspctrm'))


%figure
%plot res freq slope
for ff=1:3
subplot(2,8,13+ff)
hold on

for c=1:numel(condition)
[pks,locs]=findpeaks(log10(neg_mean_pow{ff,c}));
plot((freq_cond.freq),log10(neg_mean_pow{ff,c}),freq_cond.freq(locs),pks,'kx');

fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');
fit_foi=allfit_foi(ff,:);
xlabel(fit_filename)

end

ylabel('log10 pow')
% for each fit add cond t test
switch sel_con
    case 'main_sme'
      cond1=((neg_all_pow{ff,1}+neg_all_pow{ff,3}).*0.5);
      cond2=((neg_all_pow{ff,2}+neg_all_pow{ff,4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);

[h,p]=ttest(cond1,cond2)
plot(freq_cond.freq(h==1),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

end

title(strcat('meg neg ',sel_con,sel_freq,specific,'residual freq'))

else
 sel_all_neg=[];   

end
savefig(fig,fullfile(path_fig,strcat('suppfig6',sel_con,sel_freq,specific)))
close all
end
end

keep project_path
%%%%%%%%%%%%%%%%%

%% ieeg
% 
% 
%  path_in=fullfile(project_path,'ieeg_data');
%   path_stats=fullfile(path_in,'freq','stats');
%  path_sourcemodel=fullfile(project_path,'ieeg_data','sourcemodels');
%  path_out=fullfile(project_path,'ieeg_data','freq_tiltfit');
%  path_out_z=fullfile(project_path,'ieeg_data','freq_zinfo');
% mkdir(path_out)
% mkdir(path_out_z)
% 
%  pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
% cond={'words','faces'};
% sme={'hit','miss'};
% %load allstat for elec definition (no resected, less than 10 trials)
%  load(fullfile(path_stats,'allelecstatcondalphabeta8to20Hz300to1500ms.mat'))
% for n=1:numel(pat)
%   
% for c=1:2
%     
%     
% nan_def=find(isnan(allstat{n}.h));  
% sel_elec=allstat{n}.elecs;
% sel_elec(nan_def)=[];
% 
% load(fullfile(path_in,strcat(pat{n},'_',cond{c})))
% 
% 
%           cfg=[];
%           cfg.method = 'wavelet';
%           cfg.width = 5;
%           cfg.output     = 'pow';
%           cfg.foi = 2:1:100;
%             cfg.channel=sel_elec;
%           cfg.toi = -0.5:0.03:1.5;
%           cfg.padtype  =     'mirror';
%           cfg.pad        = 10;
%           %cfg.trials=1:3;
%           cfg.keeptrials = 'yes';
%           freq = ft_freqanalysis(cfg, data);
%                    
%           cfg=[];
%           cfg.time=[-0.5 1.5]
%           freq_z=z_trans_TF_seltime(cfg,freq);
%         z_trans_info.mean=freq_z.cfg.mean;
%          z_trans_info.std=freq_z.cfg.std;
% 
%          clear freq_z
%          save (strcat(path_out_z,pat{n},cond{c},'_z_info'), 'z_trans_info')
% 
%                
% all_freq=freq;
%           for mem=1:2
%               if mem==1
%                trials=find(all_freq.trialinfo(:,4)>=1);
%               else
%                trials=find(all_freq.trialinfo(:,4)==0);
%               end
%                freq=all_freq;
%                 freq.powspctrm=squeeze(nanmean(all_freq.powspctrm(trials,:,:,:),1));              
%                 freq.dimord='chan_freq_time';
%                 freq.trialinfo=all_freq.trialinfo(trials,:);
% 
%                 save(strcat(path_out,pat{n},cond{c},'_',sme{mem},'raw'),'freq')
%                 clear freq trials  
%                 
%           end
%            
%          clear all_freq           
% end
% end
% 
% 
% 


%% fit averages
% path_in=fullfile(project_path,'ieeg_data','freq_tiltfit');
%  
%  condition={'words_hit','words_miss','faces_hit','faces_miss'};
%  pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
%  type='robustfit';
% foi=[30 90];
% % load averages
% for c=1:numel(condition)
% for n=1:numel(pat)
% load(strcat(path_in,pat{n}, condition{c},'raw'))
% % freq=rmfield(freq,'cfg');
% % org_freq{n,c}=freq;
% % robustfit/linfit
%          cfg.toi=[-0.5,1.5];
%          cfg.fit_type=type;
%          cfg.freq2fit= foi;
%          [freq]=sh_subtr1of(cfg,freq);
%          all_freq{n,c}=freq;
% 
% end
% end
% save(strcat(path_in,'\GA_',type,'_fit',num2str(foi(1)),'to',num2str(foi(2)),'Hz'), 'all_freq','condition')
% save(strcat(path_in,'\GA_raw'), 'org_freq', 'condition')
%
%% get condition average over all electrodes

%
%get condition average for electrodes based on allstat
path_in=fullfile(project_path,'ieeg_data','freq_tiltfit');
path_fig=fullfile(project_path,'figures');
 pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
path_stats=fullfile(project_path,'ieeg_data','freq','stats');

% load  w/f rois (sourcestats...)
% define allstat file to load
freqs={'theta','alphabeta','gamma'};
freq_def=[2,5;8 20;50 90];
time_def=[1, 1.5;0.3, 1.5;0.3 1];
all_con={'mem'};
type='robustfit';

for con=1;
sel_con=all_con{con};
for f=1:3;
sel_freq=freqs{f};

foi=freq_def(f,:);
toi=time_def(f,:);
freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms'); 
file_stats=strcat('allelecstat', sel_con,sel_freq,freqstr,timestr);
load(fullfile(path_stats,file_stats))

% define which fit file
allfit_foi=[2,30;30,90;2,90];


for ff=1:size(allfit_foi,1)
fit_foi=allfit_foi(ff,:);
load(fullfile(path_in,'GA_raw'))

specific=strcat('_',type);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');
load(fullfile(path_in,fit_filename))


for n=1:numel(pat)
% define roi electrodes,  define pos/neg cluster
pos_cluster_def_all{n}=allstat{n}.h==1&allstat{n}.f>0;
neg_cluster_def_all{n}=allstat{n}.h==1&allstat{n}.f<0;
  nan_def=find(isnan(allstat{n}.h));  
% delete all nan channels  
  pos_cluster_def_all{n}(nan_def)=[];
  neg_cluster_def_all{n}(nan_def)=[];
  
for c=1:numel(condition)

freq_all{n,c}.pow=all_freq{n,c}.powspctrm;
freq_all{n,c}.offset=all_freq{n,c}.offset;
freq_all{n,c}.slope=all_freq{n,c}.slope;
freq_all{n,c}.org_pow=org_freq{n,c}.powspctrm; 
freq.freq=all_freq{1,1}.freq;
freq.time=all_freq{1,1}.time;

fit_ind(ff,1)=nearest(freq.freq,allfit_foi(ff,1));
fit_ind(ff,2)=nearest(freq.freq,allfit_foi(ff,2));

% calculate not corrected power:
        for sens=1:size(freq_all{n,c}.pow,1)
            for t=1:size(freq_all{n,c}.pow,3)
        freq_all{n,c}.linfit(sens,:,t)=[ones(size(freq.freq));log10(freq.freq)]'*[ freq_all{n,c}.offset(sens,t); freq_all{n,c}.slope(sens,t)];
            end
        end      
    end
end

% plot powerspctrm, plot slope, res freq
% plot
t1=nearest(freq.time,toi(1));
t2=nearest(freq.time,toi(2));
f1=nearest(freq.freq,foi(1));
f2=nearest(freq.freq,foi(2));
% reorganize all electrodes in one struct
for c=1:numel(condition)
    for n=1:numel(pat)
    tmp_slope{n}=freq_all{n,c}.slope;
    tmp_offset{n}=freq_all{n,c}.offset;
    tmp_linfit{n}=freq_all{n,c}.linfit;
    tmp_pow{n}=freq_all{n,c}.pow;
    tmp_org_pow{n}=freq_all{n,c}.org_pow;

    end
      slope{ff,c}= vertcat(tmp_slope{:});
      pow{ff,c}= vertcat(tmp_pow{:});
      org_pow{c}= vertcat(tmp_org_pow{:});
      linfit{ff,c}= vertcat(tmp_linfit{:});
      offset{ff,c}= vertcat(tmp_offset{:});

end

% delete all nan channels
pos_cluster_def=[pos_cluster_def_all{:}];
neg_cluster_def=[neg_cluster_def_all{:}];



if sum (pos_cluster_def)>0   
  %  pos_cluster_def(437)=0;
for c=1:numel(condition)
pos_mean_slope{ff,c}=squeeze(nanmean(nanmean(slope{ff,c}(pos_cluster_def,t1:t2),1),2));
pos_mean_offset{ff,c}=squeeze(nanmean(nanmean(offset{ff,c}(pos_cluster_def,t1:t2),1),2));
pos_mean_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(pos_cluster_def,:,t1:t2),1),3));
pos_mean_pow{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(pos_cluster_def,:,t1:t2),1),3));
pos_mean_linfit{ff,c}=squeeze(nanmean(nanmean(linfit{ff,c}(pos_cluster_def,:,t1:t2),1),3));
pos_mean_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(pos_cluster_def,f1:f2,t1:t2),1),2),3));
pos_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(pos_cluster_def,f1:f2,t1:t2),1),2),3));

pos_all_slope{ff,c}=squeeze(nanmean(slope{ff,c}(pos_cluster_def,t1:t2),2));
pos_all_offset{ff,c}=squeeze(nanmean(offset{ff,c}(pos_cluster_def,t1:t2),2));
pos_all_orgpow{c}=squeeze(nanmean(org_pow{c}(pos_cluster_def,:,t1:t2),3));
pos_all_pow{ff,c}=squeeze(nanmean(pow{ff,c}(pos_cluster_def,:,t1:t2),3));
pos_all_linfit{ff,c}=squeeze(nanmean(linfit{ff,c}(pos_cluster_def,:,t1:t2),3));
pos_all_powmean{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(pos_cluster_def,f1:f2,t1:t2),2),3));
pos_all_meanorgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(pos_cluster_def,f1:f2,t1:t2),2),3));
end
end

if sum (neg_cluster_def)>0
for c=1:numel(condition)
neg_mean_slope{ff,c}=squeeze(nanmean(nanmean(slope{ff,c}(neg_cluster_def,t1:t2),1),2));
neg_mean_offset{ff,c}=squeeze(nanmean(nanmean(offset{ff,c}(neg_cluster_def,t1:t2),1),2));
neg_mean_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(neg_cluster_def,:,t1:t2),1),3));
neg_mean_pow{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(neg_cluster_def,:,t1:t2),1),3));
neg_mean_linfit{ff,c}=squeeze(nanmean(nanmean(linfit{ff,c}(neg_cluster_def,:,t1:t2),1),3));
neg_mean_powmean{ff,c}=squeeze(nanmean(nanmean(nanmean(pow{ff,c}(neg_cluster_def,f1:f2,t1:t2),1),2),3));
neg_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(neg_cluster_def,f1:f2,t1:t2),1),2),3));

neg_all_slope{ff,c}=squeeze(nanmean(slope{ff,c}(neg_cluster_def,t1:t2),2));
neg_all_offset{ff,c}=squeeze(nanmean(offset{ff,c}(neg_cluster_def,t1:t2),2));
neg_all_orgpow{c}=squeeze(nanmean(org_pow{c}(neg_cluster_def,:,t1:t2),3));
neg_all_pow{ff,c}=squeeze(nanmean(pow{ff,c}(neg_cluster_def,:,t1:t2),3));
neg_all_linfit{ff,c}=squeeze(nanmean(linfit{ff,c}(neg_cluster_def,:,t1:t2),3));
neg_all_powmean{ff,c}=squeeze(nanmean(nanmean(pow{ff,c}(neg_cluster_def,f1:f2,t1:t2),2),3));
neg_all_meanorgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(neg_cluster_def,f1:f2,t1:t2),2),3));
end

end
end

% fig pos cluster

% bargraph slopes
fig=figure

switch sel_freq
    case 'gamma'
    
for ff=1:3
subplot(2,8,ff)
bar([pos_mean_slope{ff,:}])
sel_all=[pos_all_slope{ff,:}];
hold on 
scatter(reshape(repmat(1:4,size(sel_all,1),1),[],1) ,reshape(sel_all,[],1),'ko');
ylim([-5,0])
h=gca;
fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)
h.XTickLabel=condition;
end
title(strcat('poscluster:','allstat_',sel_con,sel_freq, num2str(foi(1)),'to', num2str(foi(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))


sel_all_pos=[[pos_all_slope{1,:}],[pos_all_slope{2,:}],[pos_all_slope{3,:}]];

% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
% plot original power plus different slopes
subplot(2,8,4:5)
for c=1:numel(condition)
plot(log10(freq.freq),log10(pos_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on


% plot slope lines for the different fits (only for fitted freqs)
for ff=1:3
   
plot(log10(freq.freq(fit_ind(ff,1):fit_ind(ff,2))),(pos_mean_linfit{ff,c}(fit_ind(ff,1):fit_ind(ff,2))),line_def{ff},'Color',color_def(c,:))
end
end
legend(reshape(repmat(condition,4,1),[],1))

switch sel_con
    case 'mem'
      cond1=((pos_all_orgpow{1}+pos_all_orgpow{3}).*0.5);
      cond2=((pos_all_orgpow{2}+pos_all_orgpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot((log10(freq.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot((log10(freq.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')


xlabel('log10(freq)')
ylabel('log10(power)')

%title(strcat('ieeg pos ',sel_con,sel_freq,specific,'raw powspctrm'))


%figure
%plot res freq slope
for ff=1:3
subplot(2,8,5+ff)
hold on

for c=1:numel(condition)

[pks,locs]=findpeaks(log10(pos_mean_pow{ff,c}));
plot((freq.freq),log10(pos_mean_pow{ff,c}),freq.freq(locs),pks,'kx');
ind(c)=locs(1);
ind(c)=locs(1);

fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)

end
ylabel('log10 pow')

% for each fit add cond t test
switch sel_con
    case 'mem'
      cond1=((pos_all_pow{ff,1}+pos_all_pow{ff,3}).*0.5);
      cond2=((pos_all_pow{ff,2}+pos_all_pow{ff,4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);

[h,p]=ttest(cond1,cond2);
plot(freq.freq(h==1),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')


end

title(strcat('ieeg pos ',sel_con,sel_freq,specific,'residual freq'))

%%%%%%%%%%%%%% fig neg cluster
    otherwise
% bargraph slopes
for ff=1:3
subplot(2,8,ff+8)
bar([neg_mean_slope{ff,:}])
sel_all=[neg_all_slope{ff,:}];
hold on 
scatter(reshape(repmat(1:4,size(sel_all,1),1),[],1) ,reshape(sel_all,[],1),'ko');
ylim([-5,0])
h=gca;
fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)
h.XTickLabel=condition;
end
title(strcat('negcluster:','allstat_',sel_con,sel_freq, num2str(foi(1)),'to', num2str(foi(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))


sel_all_neg=[[neg_all_slope{1,:}],[neg_all_slope{2,:}],[neg_all_slope{3,:}]];

% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
% plot original power plus different slopes
subplot(2,8,12:13)
for c=1:numel(condition)
plot(log10(freq.freq),log10(neg_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on
% plot slope lines for the different fits (only for fitted freqs)
for ff=1:3
   
plot(log10(freq.freq(fit_ind(ff,1):fit_ind(ff,2))),(neg_mean_linfit{ff,c}(fit_ind(ff,1):fit_ind(ff,2))),line_def{ff},'Color',color_def(c,:))
end
end
xlabel('log10(freq)')
ylabel('log10(power)')

legend(reshape(repmat(condition,4,1),[],1))
switch sel_con
    case 'mem'

      cond1=((neg_all_orgpow{1}+neg_all_orgpow{3}).*0.5);
      cond2=((neg_all_orgpow{2}+neg_all_orgpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot((log10(freq.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot((log10(freq.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')


title(strcat('ieeg neg ',sel_con,sel_freq,specific,'raw powspctrm'))


%figure
%plot res freq slope
for ff=1:3
subplot(2,8,13+ff)
hold on

for c=1:numel(condition)

[pks,locs]=findpeaks(log10(neg_mean_pow{ff,c}));
plot((freq.freq),log10(neg_mean_pow{ff,c}),freq.freq(locs),pks,'kx');
ind(c)=locs(1);

fit_foi=allfit_foi(ff,:);
fit_filename=strcat('GA_',type,'_fit',num2str(fit_foi(1)),'to',num2str(fit_foi(2)),'Hz');

xlabel(fit_filename)

end
ylabel('log10 pow')


% for each fit add cond t test
switch sel_con
    case 'mem'
      cond1=((neg_all_pow{ff,1}+neg_all_pow{ff,3}).*0.5);
      cond2=((neg_all_pow{ff,2}+neg_all_pow{ff,4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);

[h,~]=ttest(cond1,cond2)
plot(freq.freq(h==1),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

end

title(strcat('ieeg neg ',sel_con,sel_freq,specific,'residual freq'))
end
savefig(fig,fullfile(path_fig,strcat('suppfig7',sel_con,sel_freq,specific)))

close all
end
end
