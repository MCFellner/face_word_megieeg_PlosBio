project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%% get single trial source data


% path_in=fullfile(project_path,'meg_data');
% path_sourcemodel=fullfile(project_path,'meg_data','sourcemodels');
% path_out=fullfile(project_path,'meg_data','source','AUC_for_timing');
% mkdir(path_out)
% 
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%         'vp31';'vp32'};
%  
%  cond={'words', 'faces'};    
%  
% freq_bands={'theta','gamma'};
% toi=[0 1.5];
% 
% for f=1:numel(freq_bands)
%     freq_band=freq_bands{f}
%     
%     switch freq_band
%         case 'theta'
%             foi=[2 5];
%             filt_set='no'
%             filt_freq=[];
% 
%         case 'gamma'
%              foi=[50 90];
%             filt_set='yes'
%             filt_freq=30; 
%             
%     end
% 
%  for c=1:numel(cond)
%  for n=1:numel(sub)
%     
%      load(fullfile(path_in,strcat(sub{n},'_',cond{c})));    
%      load(fullfile(path_sourcemodel,strcat(sub{n},'sourcemodel10')));
% 
%    cfg=[];
% cfg.dftfilter='yes';
% cfg.dftfreq=[50 100 150];
% data=ft_preprocessing(cfg,data)
% 
% enco_trials=find( data.trialinfo(:,5)>1400);
% %covariance matrix
% 
% cfg=[];
% cfg.preproc.hpfilter=filt_set;
% cfg.preproc.hpfreq=filt_freq;
% cfg.covariance         = 'yes';
% cfg.covariancewindow   = 'all';
% cfg.keeptrials   = 'yes';
% cfg.trials=enco_trials; %encoding trials
% 
% erp=ft_timelockanalysis(cfg,data);
% 
% 
% cfg             = [];
% cfg.grid        = sourcemodel;
% cfg.vol         = vol;
% cfg.channel     = {'MEG'};
% cfg.grad        = data.grad;
% 
% sourcemodel_lf  = ft_prepare_leadfield(cfg, data);
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
% % beam every single trial
% 
% %combine all trials in a matrix (chan*(time*trials))
% trials= [data.trial{1,enco_trials}];
% 
% %combine all filters in one matrix(insidepos*chan)
% insidepos=find(sourceAll.inside);
% filters=vertcat(sourceAll.avg.filter{insidepos,:});
% 
% virtualsensors=filters*trials;
% 
% beameddata=data;
% 
% trialarray=reshape(virtualsensors,[numel(insidepos),size(data.time{1,1},2),numel(enco_trials)]);
% 
% trial=squeeze(num2cell(trialarray,[1,2]))';
% 
% beameddata.trial=trial;
% beameddata.label=cellstr(num2str(insidepos));
% beameddata.time=data.time(enco_trials);
% beameddata.trialinfo=data.trialinfo(enco_trials,:);
% beameddata=rmfield(beameddata, 'grad');
% beameddata=rmfield(beameddata, 'sampleinfo');
% 
% clear virtualsensors trials trialarray filters trial data
% 
% clear sourceAll
% clear sourcemodel sourcemodel_lf 
% 
% switch freq_band
%         case 'theta'
%           cfg=[];
%           cfg.method = 'wavelet';
%           cfg.width = 5;
%           cfg.output     = 'pow';
%           cfg.foi = foi(1):1:foi(2);
%           cfg.toi = -1:0.05:2;
%           cfg.keeptrials = 'yes';
%         case 'gamma'
%         cfg              = [];
%         cfg.foi = foi(1):1:foi(2);
%         cfg.channel      = 'all';
%         cfg.output       = 'pow';
%         cfg.pad          = 'maxperlen';
%         cfg.method      = 'mtmconvol';
%         cfg.taper       = 'dpss';
%         cfg.t_ftimwin   = ones(1,length(cfg.foi))*0.3%5; % this is a 200 ms time window, but I've gone up to 500 ms for smoother results;
%         cfg.tapsmofrq   = ones(1,length(cfg.foi))*10; 
%         cfg.keeptrials = 'yes';
%         cfg.toi = -1:0.05:2; 
% end
%            freq = ft_freqanalysis(cfg, beameddata);
%           freq_z=z_trans_TF(cfg,freq);
%           clear freq beameddata 
% 
%          
% cond_sme={'hit','miss'};
% 
%     for s= 1: numel(cond_sme)   
%     
%   condition=cond_sme{s};
%     switch condition
%             
%         case 'hit'
%             trials=find(freq_z.trialinfo(:,10)==1);
%         case 'miss'
%             trials=find(freq_z.trialinfo(:,10)==0);
%             otherwise
%     end
%     freq_cond=freq_z;
%     freq_cond.freq=mean(foi);
%     freq_cond.dimord='rpt_chan_freq_time';
%     % select toi
%     t1=nearest(freq_cond.time,toi(1));
%     t2=nearest(freq_cond.time,toi(2));
%     freq_cond.powspctrm=(nanmean(freq_z.powspctrm(trials,:,:,t1:t2),3));
%     
%     save (fullfile(path_out,strcat (sel_sub,'beamed_TF_singletrial',freq_band,condition,'_',cond{c})), 'freq_cond')
%     clear freq_cond   insidepos  trials
%     end
%  clear freq_z
% end
% end
% end
% 
% 
% %% calculate cumsum based on t values (only of negative/positive values)
% 
% 
% path_in=fullfile(project_path,'meg_data','source','AUC_for_timing');
% 
% 
% freq_bands={'theta','gamma'};
% 
% toi=[0.2 1.5];
% 
% condition={'hit_words','miss_words','hit_faces','miss_faces'};
% 
% peak_threshold=0.05;
% 
% effect='main_sme';
% 
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%        'vp11';'vp12';'vp14';'vp15';'vp18';...
%        'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%        'vp31';'vp32'};
% 
% template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));
% 
% % load stuff
% for fb=1:numel(freq_bands)
%     freq_band=freq_bands{fb};
%    
% for n=1:numel(sub)
% sel_sub=sub{n};
%         load (fullfile(path_in,strcat (sel_sub,'beamed_TF_singletrial',freq_band,condition{1})));
%         freq1=freq_cond;
%         load (fullfile(path_in,strcat (sel_sub,'beamed_TF_singletrial',freq_band,condition{1})));
%         freq2=freq_cond;
%         load (fullfile(path_in,strcat (sel_sub,'beamed_TF_singletrial',freq_band,condition{1})));
%         freq3=freq_cond;
%         load (fullfile(path_in,strcat (sel_sub,'beamed_TF_singletrial',freq_band,condition{1})));
%         freq4=freq_cond;
%         clear freq_cond
% 
% 
% % define time (there is an error in the freq.time vector
% correct_t=[0 1.5];
% corrt1=nearest(freq1.time,correct_t(1));
% corrt2=nearest(freq1.time,correct_t(2));
% freq1.time= freq1.time(corrt1:corrt2);
% 
% t1=nearest(freq1.time,toi(1));
% t2=nearest(freq1.time,toi(2));
% 
% time= freq1.time(t1:t2);
% 
% % combine conditions 
% powspctrm1=squeeze(freq1.powspctrm(:,:,:,t1:t2));
% powspctrm2=squeeze(freq2.powspctrm(:,:,:,t1:t2));
% powspctrm3=squeeze(freq3.powspctrm(:,:,:,t1:t2));
% powspctrm4=squeeze(freq4.powspctrm(:,:,:,t1:t2));
% 
% freq=[powspctrm1;powspctrm2;powspctrm3;powspctrm4];
% 
% 
% % define conditions
% hitind1=size(powspctrm1,1);
% missind1=size(powspctrm2,1);
% hitind2=size(powspctrm3,1);
% missind2=size(powspctrm4,1);
% 
% 
% mem=[repmat({'hit'},hitind1,1);repmat({'miss'},missind1,1);repmat({'hit'},hitind2,1);repmat({'miss'},missind2,1)];
% cond=[repmat({'word'},hitind1+missind1,1);repmat({'face'},hitind2+missind2,1)];
%   
% memind=strcmp('hit',mem);
% for chan=1:size(powspctrm1,2)
%     % channel loop
%     [h,p,~,stats] = ttest2(freq(memind==1,chan,:),freq(memind==0,chan,:));               
%     mem_t(chan,:)=squeeze(stats.tstat);
%     mem_p(chan,:)=squeeze(p);
%  end
% %     
% 
% % direction of effect
% 
% mean_hit1=squeeze(nanmean(powspctrm1));
% mean_hit2=squeeze(nanmean(powspctrm3));
% mean_miss1=squeeze(nanmean(powspctrm2));
% mean_miss2=squeeze(nanmean(powspctrm4));
% 
% sme=((mean_hit1+mean_hit2).*0.5)-((mean_miss1+mean_miss2).*0.5);
% 
% direction=sign(sme);
% dir_f=mem_t;
% 
% neg_cumsum=cumsum(dir_f.*(dir_f<0),2);
% pos_cumsum=cumsum(dir_f.*(dir_f>0),2);
% 
% neg_cumsum_sig=cumsum(dir_f.*(dir_f<0).*(mem_p<0.05),2);
% pos_cumsum_sig=cumsum(dir_f.*(dir_f>0).*(mem_p<0.05),2);
% 
% [sorted_negsum, ind_negsum]=sort(neg_cumsum(:,end));
% [sorted_possum, ind_possum]=sort(pos_cumsum(:,end));
%     
% peaks_neg=ind_negsum(1:(round(numel(ind_negsum)*peak_threshold)));
% peaks_pos=ind_possum(numel(ind_possum)+1-(round(numel(ind_possum)*peak_threshold)):end); 
% 
% auc_curve_neg=nanmean(neg_cumsum(peaks_neg,:))./nanmean(neg_cumsum(peaks_neg,end));
% auc_curve_pos=nanmean(pos_cumsum(peaks_pos,:))./nanmean(pos_cumsum(peaks_pos,end));
% 
% sme_negpeak=sme(peaks_neg,:);
% sme_pospeak=sme(peaks_pos,:);
% 
% 
% AUC.direction=direction;
% AUC.mem_f=mem_t;
% AUC.mem_p=mem_p;
% AUC.negcumsum=neg_cumsum;
% AUC.negcumsum_sig=neg_cumsum_sig;
% AUC.peaks_neg=peaks_neg;
% AUC.auc_curve_neg=auc_curve_neg;
% 
% AUC.poscumsum=pos_cumsum;
% AUC.poscumsum_sig=pos_cumsum_sig;
% AUC.peaks_pos=peaks_pos;
% AUC.auc_curve_pos=auc_curve_pos;
% AUC.sme=sme;
% 
% save(fullfile(path_in,strcat(freq_band,sel_sub,'_auc_info_ttestdef','toi',num2str(toi(1)*1000),'to',num2str(toi(2)*1000),'thresholdpercent',num2str((peak_threshold*100)))),'AUC')
%   clear mem_f mem_p neg_cumsum neg_cumsum_sig peaks_neg auc_curve_neg pos_cumsum pos_cumsum_sig peaks_pos auc_curve_pos sme
% 
% end
% end




%% permutations-stats for gamma vs theta

 path_in=fullfile(project_path,'meg_data','source','AUC_for_timing');
 path_figs=fullfile(project_path,'figures');

freq_bands={'theta','gamma'};

toi=[0.2 1.5];

peak_threshold=0.05;


vp={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
nrand=1000;
all_methods={'avgall_nothreshold','avgall_threshold','avgpeaks_nothreshold','avgpeaks_threshold'};
fig_names={'suppfig9A','suppfig9B','suppfig9C','fig5H'};
for m=1:numel(all_methods)
    selected_method=all_methods{m};
    fig_name=fig_names{m};
for f=1:numel(freq_bands)
    freq_band=freq_bands{f};
   
for n=1:numel(vp)
    sel_sub=vp{n};
 selected_method=all_methods{m};
 load(fullfile(path_in,strcat(freq_band,sel_sub,'_auc_info_ttestdef','toi',num2str(toi(1)*1000),'to',num2str(toi(2)*1000),'thresholdpercent',num2str((peak_threshold*100)))))


max_t=max(AUC.mem_f,[],2);
min_t=min(AUC.mem_f,[],2);
switch selected_method
case 'avgall_nothreshold'

tmp_pos=nanmean(cumsum(AUC.mem_f-repmat(min_t,1,size(AUC.mem_f,2)),2));
tmp_neg=nanmean(cumsum(AUC.mem_f-repmat(max_t,1,size(AUC.mem_f,2)),2));
peak_auc{f}.pos(n,:)=tmp_pos./tmp_pos(end);
peak_auc{f}.neg(n,:)=tmp_neg./tmp_neg(end);

peak_sme{f}.pos(n,:)=nanmean(AUC.sme);
peak_sme{f}.neg(n,:)=nanmean(AUC.sme);



case 'avgpeaks_nothreshold'
tmp_pos=(cumsum(AUC.mem_f-repmat(min_t,1,size(AUC.mem_f,2)),2));
tmp_neg=(cumsum(AUC.mem_f-repmat(max_t,1,size(AUC.mem_f,2)),2));
peak_auc{f}.pos(n,:)=nanmean(tmp_pos(AUC.peaks_pos,:))./nanmean(tmp_pos(AUC.peaks_pos,end));
peak_auc{f}.neg(n,:)=nanmean(tmp_neg(AUC.peaks_neg,:))./nanmean(tmp_neg(AUC.peaks_neg,end));

peak_sme{f}.pos(n,:)=nanmean(AUC.sme(AUC.peaks_pos,:));
peak_sme{f}.neg(n,:)=nanmean(AUC.sme(AUC.peaks_neg,:));

case 'avgall_threshold'
peak_auc{f}.pos(n,:)=nanmean(AUC.poscumsum)./nanmean(AUC.poscumsum(:,end));
peak_auc{f}.neg(n,:)=nanmean(AUC.negcumsum)./nanmean(AUC.negcumsum(:,end));
peak_sme{f}.pos(n,:)=nanmean(AUC.sme(AUC.peaks_pos,:));
peak_sme{f}.neg(n,:)=nanmean(AUC.sme(AUC.peaks_neg,:));


case 'avgpeaks_threshold'
peak_auc{f}.pos(n,:)=AUC.auc_curve_pos;
peak_auc{f}.neg(n,:)=AUC.auc_curve_neg;
peak_sme{f}.pos(n,:)=nanmean(AUC.sme(AUC.peaks_pos,:));
peak_sme{f}.neg(n,:)=nanmean(AUC.sme(AUC.peaks_neg,:));

end


end
end



% auc & sme in peaks
mean_auc_theta=nanmean(peak_auc{1}.neg);
mean_auc_gamma=nanmean(peak_auc{2}.pos);

mean_sme_theta=nanmean(peak_sme{1}.neg);
mean_sme_gamma=nanmean(peak_sme{2}.pos);

[h1,p1,~,stat1]=ttest(squeeze(peak_sme{1}.neg));
[h2,p2,~,stat2]=ttest(squeeze(peak_sme{2}.pos));

t_sme_theta=squeeze(stat1.tstat);
t_sme_gamma=squeeze(stat2.tstat);  
time=toi(1):0.05:toi(2);

data=peak_auc{1}.neg-peak_auc{2}.pos;

[h_data,p_data,~,stats_data]=ttest(data);
tsum_data=nansum(stats_data.tstat(bwlabel(h_data)==1));

% rand: switch -/+ in data

for r=1:nrand

rand_sign=sign(rand(size(data,1),1)-0.5);
data_rand=data.*repmat(rand_sign,1,size(data,2));

[h_rand,p_rand,~,stats_rand]=ttest(data_rand);

h_rand_neg=h_rand.*(stats_rand.tstat<0);
tsum_neg(r)=nansum(stats_rand.tstat(bwlabel(h_rand_neg)==1));

h_rand_pos=h_rand.*(stats_rand.tstat>0);
tsum_pos(r)=nansum(stats_rand.tstat(bwlabel(h_rand_pos)==1));
end


dist_pos=sort(tsum_pos,'descend');
dist_neg=sort(tsum_neg,'ascend');

p_pos=(nrand-sum(dist_pos<tsum_data))./nrand;
p_neg=(nrand-sum(dist_neg>tsum_data))./nrand;

if tsum_data<0
    pcorr=p_neg;
elseif tsum_data>= 0
    pcorr=p_pos;
end

% info for sig cluster
cluster_ind=find(bwlabel(h_data==1));


fig=figure
subplot(2,1,1)
hold on
rectangle('Position', [time(cluster_ind(1)),0,time(cluster_ind(end)-cluster_ind(1)),1],'Facecolor',[0.9 0.9 0.9]); 
text(time(cluster_ind(1)),1,strcat('pcorr=',num2str(pcorr)));

yyaxis left
plot(time,mean_auc_theta)
plot(time,mean_auc_gamma)

yyaxis right
plot(time, p_data)
legend('theta','gamma')
title('percent AUC')


subplot(2,1,2)
hold on
plot(time,t_sme_theta)
plot(time,t_sme_gamma)
legend('theta','gamma')
title('SME')

savefig(fig,fullfile(path_figs,fig_name))
close all
end





%%%%%%%%%%%%%%%%%%%%%%%
%% time sliding cluster stat

path_in=fullfile(project_path,'meg_data','source');
path_out=fullfile(project_path,'meg_data','source','stats','timeslide');
mkdir(path_out)
cd(path_in)

tois=-0.2:0.1:1.4;   
win=0.300;

sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
conditions={'hit_words','miss_words','hit_faces','miss_faces'};

% define all contrasts of interest
freq_def={'theta','gamma'};

template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));

for f=1:numel(freq_def)
    freq_band=freq_def{f};
switch freq_band
    case 'theta'
    effects={'main_sme'};
    foi=[2 5];
    sel_freq_band='lf';
    case 'gamma'
    effects={'main_sme'};
    foi=[50 90];
    sel_freq_band='hf';        
end

for e=1:numel(effects)
    effect=effects{e};
    % load data and construct source grandaverage files
    

for t_win=1:numel(tois)
toi=[tois(t_win),tois(t_win)+win];
       

for n=1:numel(sub)
        load (fullfile(path_in,strcat(sub{n},'template_source'))) %load a template source, dummy for later stats and plotting
        sel_sub=sub{n};
          switch effect
            case 'main_sme'
            load (strcat(sel_sub,'beamed_TF_',sel_freq_band, conditions{1}));
            freq3=freq_cond;
            load (strcat(sel_sub,'beamed_TF_', sel_freq_band,conditions{3}));
            freq4=freq_cond;
            freq1=freq_cond;
            freq1.powspctrm=(freq3.powspctrm+freq4.powspctrm)./2;

            load (strcat(sel_sub,'beamed_TF_', sel_freq_band,conditions{2}));
            freq5=freq_cond;
            load (strcat(sel_sub,'beamed_TF_', sel_freq_band,conditions{4}));
            freq6=freq_cond;
            freq2=freq_cond;
            freq2.powspctrm=(freq5.powspctrm+freq6.powspctrm)./2;
            clear freq_cond freq5 freq6 freq3 freq4               
              otherwise
          end
        
        
t1=nearest(freq1.time,toi(1));
t2=nearest(freq1.time,toi(2));

f1=nearest(freq1.freq,foi(1));
f2=nearest(freq1.freq,foi(2));

pow1=squeeze(nanmean(nanmean((freq1.powspctrm(:,f1:f2,t1:t2)),2),3));
pow2=squeeze(nanmean(nanmean((freq2.powspctrm(:,f1:f2,t1:t2)),2),3));

sourceAll.pos = template.sourcemodel.pos;
sourceAll.dim = template.sourcemodel.dim;

insidepos=find(sourceAll.inside);

source1=sourceAll;
source1.avg.pow(insidepos)=pow1;
source2=sourceAll;
source2.avg.pow(insidepos)=pow2;

sourcega1{n}=source1;
sourcega2{n}=source2;

clear freq1 freq2 insidepos pow1 pow2 source1 source2 
end

%
design(1, :) = repmat(1:length(sub), 1, 2);
design(2, :) = [ones(1, length(sub)) ones(1, length(sub)) * 2];

cfg = [];
cfg.dim         = sourcega1{1}.dim;
%cfg.method      = 'analytic';
cfg.statistic   = 'depsamplesT';
cfg.parameter   = 'avg.pow';
cfg.correctm    = 'cluster';
cfg.method='montecarlo';
cfg.numrandomization = 1000;
cfg.alpha       = 0.05;
cfg.clusteralpha = 0.05;
cfg.tail        = 0;
cfg.design = design;
cfg.ivar = 2;% indepva1 
cfg.uvar = 1;% units of observation  should not exist in indepsamp test    

stat = ft_sourcestatistics(cfg,sourcega1{:},sourcega2{:});



  freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');

file_name=fullfile(path_out,strcat('stat_',effect,freqstr,timestr));
save(file_name,'stat');

end
end
end
%%
% timecourse smes

% load stat files
path_stat=fullfile(project_path,'meg_data','source','stats','timeslide');
 path_figs=fullfile(project_path,'figures');


freq_band={'theta','gamma'};


effects={'main_sme'};%{'interaction';'main_sme';'word_sme';'main_cond';'face_sme'};
c_alpha=0.05;
tois=-0.1:0.1:1.4;   
win=0.300;

for f=1: numel(freq_band)
    sel_freq=freq_band{f};
switch sel_freq
    case 'theta'
    foi=[2 5];
    case 'gamma'
    foi=[50 90];
end
    
for e=1:numel(effects)
    effect=effects{e}

for t_win=1:numel(tois)
toi=[tois(t_win),tois(t_win)+win];
  freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');

file_name=fullfile(path_stat,strcat('stat_',effect,freqstr,timestr));
load(file_name);

% get sig for biggest neg/pos cluster plus significancs
if isfield(stat,'posclusters')
    if ~isempty(stat.posclusters)
    pos_tsum(t_win,e,f)=stat.posclusters(1).clusterstat;
    pos_p(t_win,e,f)=stat.posclusters(1).prob; % if more than one cluster sum t_values
    
    %more_clust
    else
        pos_tsum(t_win,e,f)=NaN;
        pos_p(t_win,e,f)=NaN;
    end
end

if isfield(stat,'negclusters')
    if ~isempty(stat.negclusters)
    neg_tsum(t_win,e,f)=stat.negclusters(1).clusterstat;
    neg_p(t_win,e,f)=stat.negclusters(1).prob;
    else
        neg_tsum(t_win,e,f)=NaN;
        neg_p(t_win,e,f)=NaN;
    end
end
all_tsum(t_win,e,f)=nansum(stat.stat);
allpos_tsum(t_win,e,f)=nansum(stat.stat.*(stat.stat>stat.cfg.clustercritval(2)));
allneg_tsum(t_win,e,f)=nansum(stat.stat.*(stat.stat<stat.cfg.clustercritval(1)));
end
end
end

%
% fdr correct p-values

neg_p_sel=neg_p(4:end,:,:);
    % vectorize p-values
    pvals_neg=reshape(neg_p_sel,1,[]);
    % fdr 
    [pthr,pcor,padj] = fdr(pvals_neg);
    neg_pcorr=reshape(padj,size(neg_p_sel));
    
    pos_p_sel=pos_p(4:end,:,:);
   
    % vectorize p-values
    pvals_pos=reshape(pos_p_sel,1,[]);
    % fdr 
    [pthr,pcor,padj] = fdr(pvals_pos);
    pos_pcorr=reshape(padj,size(pos_p_sel));    
    
    
toplot(:,:,1:2)=allneg_tsum(4:end,:,1:2);    
toplot(:,:,3)=allneg_tsum(4:end,:,3);    

toi_plot=tois(4:end);


fig=figure
bar(squeeze(toplot))
legend(freq_band)
title(strcat('neg cluster t-sum:', freq_band{1:2},'pos cluster t-sum:', freq_band{3}))
ylabel('cluster tsum source')
fig=gca;
fig.XTick=1:numel(toi_plot);
fig.XTickLabel=toi_plot;
hold on
for t=1:numel(toi_plot)
    if neg_p_sel(t,1,1)<0.05%neg_pcorr(t,1,1)<0.05
    text(t+0.2,0,'*','Color','blue','FontSize',40)
    else    
    end
    if neg_p_sel(t,1,1)<0.05%neg_pcorr(t,1,2)<0.05
    text(t+0,0,'*','Color','green','FontSize',40)
    else    
    end
    if pos_p_sel(t,1,3)<0.05%pos_pcorr(t,1,3)<0.05
    text(t+0.2,0,'*','Color',[0.9 0.9 0.3],'FontSize',40)
    else    
    end
end

savefig(fig,fullfile(path_figs,'fig5G'))
close all