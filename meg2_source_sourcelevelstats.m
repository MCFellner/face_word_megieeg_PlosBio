project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20150421\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));

%% source routine
% files generated here are provided in the file download

% 
% %covariance matrix based on all trials
% % 
% path_in=fullfile(project_path,'meg_data');
% path_sourcemodel=fullfile(project_path,'meg_data','sourcemodels');
% path_out=fullfile(project_path,'meg_data','source');
% mkdir(path_out)
% 
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%         'vp31';'vp32'};
%  
% cond={'words', 'faces'};    
% 
% cd(path_out)
% freq_bands={'lf','hf'};
% for f=1:numel(freq_bands)
%     sel_freq_band=freq_bands{f};
%     
% for c=1:numel(cond)
% for n=1:numel(sub)
%     load(fullfile(path_in,strcat(sub{n},'_',cond{c})));    
%     load(fullfile(path_sourcemodel,strcat(sub{n},'sourcemodel10')))
%     
%     cfg=[];
%     cfg.dftfilter='yes';
%     data=ft_preprocessing(cfg,data)  ;
%     
%     sel_trials=find(data.trialinfo(:,5)>1400);
%     %covariance matrix
%     cfg=[];
%     switch sel_freq_band
%         case 'hf'
%     cfg.preproc.hpfilter='yes';
%     cfg.preproc.hpfreq=30;        
%     end
%     cfg.covariance         = 'yes';
%     cfg.covariancewindow   = 'all';
%     cfg.trials=sel_trials;
%     cfg.keeptrials   = 'yes';
%     cfg.trials=sel_trials; %encoding trials
%     erp=ft_timelockanalysis(cfg,data);
% 
% % LCMV spatial filter
%     cfg             = [];
%     cfg.grid        = sourcemodel;
%     cfg.vol         = vol;
%     cfg.channel     = {'MEG'};
%     cfg.grad        = data.grad;
%     sourcemodel_lf  = ft_prepare_leadfield(cfg, data);
% 
%     cfg              = []; 
%     cfg.method       = 'lcmv';
%     cfg.grid = sourcemodel_lf;
%     cfg.vol          = vol;
%     cfg.lcmv.lambda       = '5%';
%     cfg.lcmv.projectnoise = 'yes';
%     cfg.lcmv.keepfilter   = 'yes';
%     cfg.lcmv.realfilter   = 'yes';
%     cfg.lcmv.fixedori   = 'yes';
%     cfg.grad        = data.grad;
%     sourceAll = ft_sourceanalysis(cfg, erp);
%     clear erp
% 
% % beam every single trial
%     %combine all trials in a matrix (chan*(time*trials))
%     trials= [data.trial{1,sel_trials}];
%     %combine all filters in one matrix(insidepos*chan)
%     insidepos=find(sourceAll.inside);
%     filters=vertcat(sourceAll.avg.filter{insidepos,:});
%     virtualsensors=filters*trials;
%     beameddata=data;
%     trialarray=reshape(virtualsensors,[numel(insidepos),size(data.time{1,1},2),numel(sel_trials)]);
%     trial=squeeze(num2cell(trialarray,[1,2]))';
%     beameddata.trial=trial;
%     beameddata.label=cellstr(num2str(insidepos));
%     beameddata.time=data.time(sel_trials);
%     beameddata.trialinfo=data.trialinfo(sel_trials,:);
%     beameddata=rmfield(beameddata, 'grad');
%     beameddata=rmfield(beameddata, 'sampleinfo');
% 
%     clear virtualsensors trials trialarray filters trial data
%     save (strcat(sub{n},'sourceAll',cond{c},sel_freq_band),'sourceAll')
% 
%     clear sourceAll
%     clear sourcemodel sourcemodel_lf 
% 
%     % tf transform on beamed single trials
%     
%      switch sel_freq_band
%         case 'lf'   
%         cfg=[];
%         cfg.method = 'wavelet';
%         cfg.width = 5;
%         cfg.output     = 'pow';
%         cfg.foi = 2:1:30;
%         cfg.toi = -0.5:0.05:1.5;
%         cfg.keeptrials = 'yes';
%         freq = ft_freqanalysis(cfg, beameddata);
%         cfg.time=[freq.time(1) freq.time(end)];
%         freq_z=z_trans_TF_seltime(cfg,freq);
%         clear freq beameddata 
%         case 'hf'
%         cfg              = [];
%         cfg.channel      = 'all';
%         cfg.output       = 'pow';
%         cfg.pad          = 'maxperlen';
%         cfg.keeptrials   = 'yes';
%         cfg.toi          = -1:0.05:2;
%         cfg.method      = 'mtmconvol';
%         cfg.taper       = 'dpss';
%         cfg.foi         = 30:5:100;
%         cfg.t_ftimwin   = ones(1,length(cfg.foi))*0.3;
%         cfg.tapsmofrq   = ones(1,length(cfg.foi))*10; 
%         cfg.trials=1:10;
%         freq = ft_freqanalysis(cfg, beameddata);
%         cfg.time=[freq.time(1) freq.time(end)];
%         freq_z=z_trans_TF_seltime(cfg,freq);
%         cfg.fwhm_t=0.2;
%         cfg.fwhm_f=10;
%         freq_z= smooth_TF(cfg,freq_z);
%         clear freq beameddata 
%      end
% 
%     % separate in      
%     cond_sme={'hit','miss'};
% 
%     for s= 1: numel(cond_sme)   
%      condition=cond_sme{s};
%     switch condition           
%         case 'hit'
%             trials=find(freq_z.trialinfo(:,4)==1);
%         case 'miss'
%             trials=find(freq_z.trialinfo(:,4)==0);
%     end        
%     freq_cond=freq_z;
%     freq_cond.dimord='chan_freq_time';
%     freq_cond.powspctrm=squeeze(nanmean(freq_z.powspctrm(trials,:,:,:),1));    
%     save (strcat(sub{n},'testbeamed_TF_',sel_freq_band,condition,'_',cond{c}), 'freq_cond')
%     clear freq_cond   
%     end    
% end
% end
% end
%% change fieldtrip version
fieldtrip_path='D:\matlab_tools\fieldtrip-20150421\';

rmpath (genpath(fieldtrip_path))
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';
addpath(fieldtrip_path)
ft_defaults
%% source level stats
path_in=fullfile(project_path,'meg_data','source');
path_out=fullfile(project_path,'meg_data','source','stats');
mkdir(path_out)
cd(path_in)

sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
conditions={'hit_words','miss_words','hit_faces','miss_faces'};

% define all contrasts of interest
freq_def={'theta','alphabeta','gamma'};

template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));

for f=1:numel(freq_def)
    sel_freq_def=freq_def{f};
switch sel_freq_def
    case 'alphabeta'
    effects={'main_cond','main_sme'};
    foi=[8 20];
    toi=[0.3 1.5];
    sel_freq_band='lf';
    case 'theta'
    effects={'main_sme'};
    foi=[2 5];
    toi=[1 1.5];
    sel_freq_band='lf';
    case 'gamma'
    effects={'main_cond','main_sme'};
    foi=[50 90];
    toi=[0.3 1];
    sel_freq_band='hf';        
end

for e=1:numel(effects)
    effect=effects{e};
    % load data and construct source grandaverage files
    for n=1:numel(sub)
        load (fullfile(path_in,strcat(sub{n},'template_source'))) %load a template source, dummy for later stats and plotting
        sel_sub=sub{n};
          switch effect
            case 'main_cond' 
            load (strcat(sel_sub,'beamed_TF_',sel_freq_band, conditions{1}));
            freq3=freq_cond;
            load (strcat(sel_sub,'beamed_TF_',sel_freq_band,conditions{2}));
            freq4=freq_cond;
            freq1=freq_cond;
            freq1.powspctrm=(freq3.powspctrm+freq4.powspctrm)./2;

            load (strcat(sel_sub,'beamed_TF_', sel_freq_band,conditions{3}));
            freq5=freq_cond;
            load (strcat(sel_sub,'beamed_TF_', sel_freq_band,conditions{4}));
            freq6=freq_cond;         
            freq2=freq_cond;
            freq2.powspctrm=(freq5.powspctrm+freq6.powspctrm)./2;
            clear freq_cond freq5 freq6 freq3 freq4

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

        % get definition of gridsensors
        sourceAll.pos = template.sourcemodel.pos;
        sourceAll.dim = template.sourcemodel.dim;
        insidepos=find(sourceAll.inside);

        % reformat channel defined data to source defined ft data
        source1=sourceAll;
        source1.avg.pow(insidepos)=pow1;
        source2=sourceAll;
        source2.avg.pow(insidepos)=pow2;

        sourcega1{n}=source1;
        sourcega2{n}=source2;
        diff_all(n,:)=source1.avg.pow-source2.avg.pow;
        clear pow1 pow2 source1 source2 freq1 freq2
    end

% design matrix
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
stat.avg=nanmean(diff_all)';

  freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');
%%% save stats 
  save(fullfile(path_out,strcat('sourcestats_freq',effect,freqstr,'time',timestr,'.mat')),'stat')
end
end

%% source based plots based on source stats 

% figure 2b and 2e (face-word alpha/beta and gamma)
%figure 4 a c e (smes plus bargraphs)
path_in=fullfile(project_path,'meg_data','source','stats');
path_figs=fullfile(project_path,'figures');

freq_def={'theta','alphabeta','gamma'};
hemisphere={'left','right'};
% stuff needed for plotting
mri_file=fullfile(project_path,'scripts','additional_scripts','T1.nii');
mri=ft_read_mri(mri_file);
surface_file{1}=fullfile(project_path,'scripts','additional_scripts','carretleft.mat');
surface_file{2}=fullfile(project_path,'scripts','additional_scripts','carretright.mat');
load(fullfile(project_path,'scripts','additional_scripts','red2blue_colormap.mat'));
views(1,:,:)=[-90,30;90 -30;-90,0;90,0;0,-90;90 -40;];
views(2,:,:)=[90,30;-90 -30;90,0;-90,0;0,-90;-90 -40];

for f=1:numel(freq_def)
    sel_freq_def=freq_def{f};
    switch sel_freq_def
        case 'alphabeta'
        effects={'main_cond','main_sme'};
        foi=[8 20];
        toi=[0.3 1.5];
        sel_freq_band='lf';
        fig_names={'fig2Bsourceplot','fig4Csourceplot'};
        case 'theta'
        effects={'main_sme'};
        foi=[2 5];
        toi=[1 1.5];
        sel_freq_band='lf';
        fig_names={'fig4Asourceplot'};

        case 'gamma'
        effects={'main_cond','main_sme'};
        foi=[50 90];
        toi=[0.3 1];
        sel_freq_band='hf';  
        fig_names={'fig2Esourceplot','fig4Esourceplot'};
    end
  freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');
  for e=1:numel(effects)
      effect=effects{e};
  load(fullfile(path_in,strcat('sourcestats_freq',effect,freqstr,'time',timestr,'.mat')))
  statmask=stat;
  statmask.stat=stat.stat.*stat.mask;
        cfg.parameter = 'stat';
        statmask=ft_sourceinterpolate(cfg, statmask,mri);
        statmask.mask=(statmask.stat<=stat.cfg.clustercritval(1) |statmask.stat>=stat.cfg.clustercritval(2));

        for h=1:numel(hemisphere)
        cfg = [];
        cfg.method         = 'surface';
        cfg.funparameter   = 'stat';
        cfg.maskparameter  = 'mask';
        cfg.funcolorlim    = [-6 6];
        cfg.funcolormap    = 'jet';
        cfg.projmethod     = 'nearest';
        cfg.surffile       = surface_file{h};
        cfg.surfdownsample = 5;
        %cfg.camlight = 'yes';
        cfg.camlight       = 'no';
        ft_sourceplot(cfg, statmask);
        material DULL

        view(squeeze(views(h,1,:))');
        c1=camlight(0,0);
        set(c1, 'style', 'infinite');

        view(squeeze(views(h,2,:))');
        c2=camlight(0, 0);
        set(c2, 'style', 'infinite');

        view(squeeze(views(h,3,:))');
        print('-f1','-r600','-dtiff',fullfile(path_figs,strcat(fig_names{e},hemisphere{h},'_lat.tiff'))) 

        view(squeeze(views(h,6,:))');
        print('-f1','-r600','-dtiff',fullfile(path_figs,strcat(fig_names{e},hemisphere{h},'_tiltedmed.tiff'))) 
        clear c1 c2 

        close all
        end  
  end
end
%% extract data from source to plot bargraphs
%figure 4 a c e (smes plus bargraphs)
path_stat=fullfile(project_path,'meg_data','source','stats');
path_in=fullfile(project_path,'meg_data','source');
path_figs=fullfile(project_path,'figures');

condition={'hit_words','miss_words','hit_faces','miss_faces'};
sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
load(fullfile(project_path,'scripts','additional_scripts','red2blue_colormap.mat'));
template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));
effect='main_sme';
freq_def={'theta','alphabeta','gamma'};
for f=1:numel(freq_def)
    sel_freq_def=freq_def{f};
switch sel_freq_def
    case 'alphabeta'
    all_cluster='neg';
    foi=[8 20];
    toi=[0.3 1.5];
    sel_freq_band='lf';
    fig_names='fig4Cbarplot';
    case 'theta'
    all_cluster='neg';
    foi=[2 5];
    toi=[1 1.5];
    sel_freq_band='lf';
    fig_names='fig4Abarplot';
    case 'gamma'
    all_cluster='pos';
    foi=[50 90];
    toi=[0.3 1];
    sel_freq_band='hf'; 
    fig_names='fig4Ebarplot';
end

   cluster=all_cluster;
   freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
   timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');   
   
   load(fullfile(path_stat,strcat('sourcestats_freq',effect,freqstr,'time',timestr,'.mat')))

% load all source data and combine in one matrix   
for n=1:numel(sub)
        load (fullfile(path_in,strcat(sub{n},'template_source'))) %load a template source, dummy for later stats and plotting
        sel_sub=sub{n};
    for c=1:numel(condition)     
        cd(path_in)
        load (strcat(sel_sub,'beamed_TF_',sel_freq_band, condition{c}));
        freq=freq_cond;
        t1=nearest(freq.time,toi(1));
        t2=nearest(freq.time,toi(2));
        f1=nearest(freq.freq,foi(1));
        f2=nearest(freq.freq,foi(2));
        pow{c}(n,:,:,:)=squeeze(nanmean(nanmean((freq.powspctrm(:,f1:f2,t1:t2)),2),3));
    end
end
    sourceAll.pos = template.sourcemodel.pos; % note: all sourcemodels are in the same space/grid
    sourceAll.dim = template.sourcemodel.dim;
 
     if strcmp(cluster,'neg')
     clust_def=stat.mask.*(stat.stat<0);
     elseif strcmp(cluster,'pos')
     clust_def=stat.mask.*(stat.stat>0);
     end
 % position indexes here are wholebrain, pow is only for inside pos
    insidepos=find(sourceAll.inside);
    clust_def=clust_def(insidepos);

%  scatter bar plots
    for c=1:numel(condition)
    pow_mean{c}=squeeze(nanmean(pow{c}(:,logical(clust_def)),2));
    end

% ttests
[h_smeword,p_smeword,~,stat_smeword]=ttest(pow_mean{1},pow_mean{2});
[h_smeface,p_smeface,~,stat_smeface]=ttest(pow_mean{3},pow_mean{4});
[h_int,p_int,~,stat_int]=ttest(pow_mean{1}-pow_mean{2},pow_mean{3}-pow_mean{4});

% barplot all 4 conditions
toplot=[mean(pow_mean{1}),mean(pow_mean{2}),mean(pow_mean{3}),mean(pow_mean{4})];
fig=figure
subplot(1,2,1)
bar(toplot)
hold on
scatter([ones(size(pow{1},1),1);ones(size(pow{1},1),1)*2;ones(size(pow{1},1),1)*3;ones(size(pow{1},1),1)*4] ,[pow_mean{1};pow_mean{2};pow_mean{3};pow_mean{4}]);
title(strcat(' cluster',num2str(foi(1)),'-',num2str(foi(2)),'Hz',num2str(toi(1)),'-',num2str(toi(2)),'sec'))
ax=gca;
ax.XTickLabel={'words hit','words miss','faces hit','faces miss'};
subplot(1,2,2)
text(0,0.5,strcat('contrast  interaction'))
text(0.1,0.45,strcat('p=', num2str(p_int),'; T(',num2str(stat_int.df),')=',num2str(stat_int.tstat)))
text(0,0.3,strcat('contrast  smeface'))
text(0.1,0.25,strcat('p=', num2str(p_smeface),'; T(',num2str(stat_smeface.df),')=',num2str(stat_smeface.tstat)))
text(0,0.1,strcat('contrast  smeword'))
text(0.1,0.05,strcat('p=', num2str(p_smeword),'; T(',num2str(stat_smeword.df),')=',num2str(stat_smeword.tstat)))
axis off
savefig(fig,fullfile(path_figs,strcat(fig_names,'.fig')))
close all
end
   
%% plot gamma in alpha/beta cluster
% figure 3a and 3c 
path_in=fullfile(project_path,'meg_data','source');
path_figs=fullfile(project_path,'figures');
path_stat=fullfile(project_path,'meg_data','source','stats');

condition={'hit_words','miss_words','hit_faces','miss_faces'};
sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
load(fullfile(project_path,'scripts','additional_scripts','red2blue_colormap.mat'));

% stat definition
    effect='main_cond';
    sel_freq_def='alphabeta'
    all_cluster={'pos','neg'};
    foi=[8 20];
    toi=[0.3 1.5];
    sel_freq_band='lf';
    freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
   timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');   
    
   load(fullfile(path_stat,strcat('sourcestats_freq',effect,freqstr,'time',timestr,'.mat')))
template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm.mat'));

    
% gamma data definition
    sel_freq_band='hf';
    fig_names={'fig3C','fig3A'};

for clus=1:numel(all_cluster)
   cluster=all_cluster{clus};
   fig_name=fig_names{clus};
   
% load all source data and combine in one matrix   
for n=1:numel(sub)
        load (fullfile(path_in,strcat(sub{n},'template_source'))) %load a template source, dummy for later stats and plotting
        sel_sub=sub{n};
        for c=1:numel(condition)     
        load (strcat(sel_sub,'beamed_TF_',sel_freq_band, condition{c}));
        freq=freq_cond;
        t1=nearest(freq.time,toi(1));
        t2=nearest(freq.time,toi(2));
        f1=nearest(freq.freq,foi(1));
        f2=nearest(freq.freq,foi(2));
        pow{c}(n,:,:,:)=squeeze(nanmean(nanmean((freq.powspctrm(:,f1:f2,t1:t2)),2),3));
    end
end
    sourceAll.pos = template.sourcemodel.pos; % note: all sourcemodels are in the same space/grid
    sourceAll.dim = template.sourcemodel.dim;
    insidepos=find(sourceAll.inside);
 
     if strcmp(cluster,'neg')
     clust_def=stat.mask.*(stat.stat<0);
     elseif strcmp(cluster,'pos')
     clust_def=stat.mask.*(stat.stat>0);
     end
 % position indexes here are wholebrain, pow is only for inside pos
    sourceAll.pos = template.sourcemodel.pos;
    sourceAll.dim = template.sourcemodel.dim;
    insidepos=find(sourceAll.inside);clust_def=clust_def(insidepos);

%  scatter bar plots
    for c=1:numel(condition)
    pow{c}=squeeze(nanmean(nanmean(nanmean(pow{c}(:,clust_def,f1:f2,t1:t2),2),3),4));
    pow_tf{c}=squeeze(nanmean(pow{c}(:,clust_def,:,:),2));
    end
    
    cond1=(pow{1}+pow{2}).*0.5;
    cond2=(pow{3}+pow{4}).*0.5;
    
    cond1_tf=(pow_tf{1}+pow_tf{2}).*0.5;
    cond2_tf=(pow_tf{3}+pow_tf{4}).*0.5;

    % ttests
    [h_mean,p_mean,~,stat_mean]=ttest(cond1,cond2);
    [h_tf,p_tf,~,stat_tf]=ttest(cond1_tf,cond2_tf);


        % barplot
        toplot=[mean(cond1),mean(cond2)];
        fig=figure;
        subplot(1,2,1)
        bar(toplot)
        hold on
        scatter([ones(size(pow{1},1),1);ones(size(pow{1},1),1)*2] ,[cond1;cond2]);
        title(strcat('cluster',num2str(foi(1)),'-',num2str(foi(2)),'Hz',num2str(toi(1)),'-',num2str(toi(2)),'sec'))
        ax=gca;
        ax.XTickLabel={'words','faces'};
        subplot(1,2,2)
        text(0,0.5,strcat('contrast  conditions'))
        text(0.1,0.45,strcat('p=', num2str(p_mean),'; T(',num2str(stat_mean.df),')=',num2str(stat_mean.tstat)))

        axis off
        savefig(fig,fullfile(path_figs,strcat(fig_name,'barplot.fig')))
        close all

        % tf plot
        fig=figure;
        imagesc(stat.time,stat.freq,squeeze(stat.stat).*-1,[-4 4])
        set(gca,'YDir','normal')
        colormap('red2blue')
        savefig(fig,fullfile(path_figs,strcat(fig_name,'tf.fig')))
        close all

end