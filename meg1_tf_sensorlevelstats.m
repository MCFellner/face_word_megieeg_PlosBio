project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));
%% freq analysis

% files generated here are provided in the download
% 
% path_in=fullfile(project_path,'meg_data');
% path_out=fullfile(path_in,'freq');
% mkdir (path_out);
% 
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%         'vp31';'vp32'};
% 
% cond={'words','faces'};
% sme={'hit','miss'};
% load(fullfile(project_path,'scripts','additional_scripts','neighbours_tri.mat'));
% freq_bands={'lf','hf'}
% for f=1%:numel(freq_bands)
%     sel_freq_band=freq_bands{f};
%     for c=1%:numel(cond)
%         for n=1:numel(sub)
%             load(fullfile(path_in,strcat(sub{n},'_',cond{c})));
% 
%             % trials with ok reaction time & dft filter for line noise
%             cfg=[];
%             cfg.dftfilter='yes'; % default setting [50,100,150]
%             cfg.trials=find(data.trialinfo(:,5)>1400);
%             data=ft_preprocessing(cfg, data)
% 
%             % calculate planar gradients
%             cfg=[];
%             cfg.neighbours=neighbours;
%             data=ft_megplanar(cfg, data);
%             
%          switch sel_freq_band
%             case 'lf'
%             cfg=[];
%             cfg.method = 'wavelet';
%             cfg.width = 5;
%             cfg.output     = 'pow';
%             cfg.foi = 1:0.5:29;
%             cfg.toi = -1.5:0.03:2.5;
%             cfg.keeptrials = 'yes';         
%             freq = ft_freqanalysis(cfg, data);
%               % combine planar gradients 
%              cfg=[];
%              cfg.combinegrad  = 'yes';
%              freq=ft_combineplanar(cfg, freq);
% 
%               cfg=[];
%               cfg.time=[-1 1.5];
%               freq_z=z_trans_TF_acrosschan(cfg,freq);
%              %smooth single trials
%              cfg=[]; 
%              cfg.fwhm_t=0.2;
%              cfg.fwhm_f=2;             
%              freq= smooth_TF(cfg,freq)
%              
%             case 'hf'
%             cfg              = [];
%             cfg.channel      = 'all';
%             cfg.output       = 'pow';
%             cfg.pad          = 'maxperlen';
%             cfg.keeptrials   = 'yes';
%              cfg.toi = -1.5:0.03:2.5;
%             cfg.method      = 'mtmconvol';
%             cfg.taper       = 'dpss';
%             cfg.foi         = 30:5:120;
%             cfg.t_ftimwin   = ones(1,length(cfg.foi))*0.3; 
%             cfg.tapsmofrq   = ones(1,length(cfg.foi))*10; 
%             freq=ft_freqanalysis(cfg,data);     
%                           % combine planar gradients 
%              cfg=[];
%              cfg.combinegrad  = 'yes';
%              freq=ft_combineplanar(cfg, freq);
%               %smooth single trials
%              cfg=[]; 
%              cfg.fwhm_t=0.2;
%              cfg.fwhm_f=10;
%              freq= smooth_TF(cfg,freq)
%              
%               cfg=[];
%               cfg.time=[-1 1.5];
%               freq_z=z_trans_TF_seltime(cfg,freq);
%          end     
% 
%               freq_noblall=freq; % no baseline data for supplemental figures
%               clear freq 
% 
%         % seperate trials in conditions and average            
%             for mem=1:2
%             trials=find(freq_z.trialinfo(:,4)==(abs(mem-2)));
%             freq=freq_z;
%             freq_nobl=freq_noblall;
%             freq.powspctrm=squeeze(nanmean(freq_z.powspctrm(trials,:,:,:),1));
%             freq.dimord='chan_freq_time';
%             freq.trialinfo=freq_z.trialinfo(trials,:);
%             
%             freq_nobl.powspctrm=squeeze(nanmean(freq_nobl.powspctrm(trials,:,:,:),1));
%             freq_nobl.dimord='chan_freq_time';
%             freq_nobl.trialinfo=freq_nobl.trialinfo(trials,:);
%             save (fullfile(path_out,strcat(sub{n},'_',sel_freq_band,'_',cond{c},'_',sme{mem},'.mat')),'freq');
%             save (fullfile(path_out,strcat(sub{n},'_',sel_freq_band,'_',cond{c},'_',sme{mem},'nobl.mat')),'freq_nobl');
% 
%             clear freq trials data
%             end
%      end
%     end
% end
% %% grandaverage (combines files for stats)
% %all subjects with more than 30 trials
% sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
%         'vp11';'vp12';'vp14';'vp15';'vp18';...
%         'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
%         'vp31';'vp32'};
% 
% path_in=fullfile(project_path,'meg_data','freq');
% cond={'words_hit','words_miss','faces_hit','faces_miss'};
% freq_bands={'lf','hf'}
% baseline_cond={'z','nobl'};
% for b=1:numel(baseline_cond)
%     sel_bl=baseline_cond{b};
% for f=1:numel(freq_bands)
%     sel_freq_band=freq_bands{f};
% for c=1:numel(cond)
%     for n=1:numel(sub)
%         switch sel_bl
%             case 'z'
%             load (fullfile(path_in,strcat(sub{n},'_',sel_freq_band,'_',cond{c})))
%              delete(fullfile(path_in,strcat(sub{n},'_',sel_freq_band,'_',cond{c},'.mat')));  % sub specific averages not needed anymore, delete them
%             freq_all{n}=freq;
%             clear freq  
%             case 'nobl'
%             load (fullfile(path_in,strcat(sub{n},'_',sel_freq_band,'_',cond{c},'nobl')))
%             delete(fullfile(path_in,strcat(sub{n},'_',sel_freq_band,'_',cond{c},'nobl.mat')));
%             freq_all{n}=freq_nobl;
%             clear freq  
%         end
%     end
%     cfg=[];
%     cfg.keepindividual = 'yes';
%     freq=ft_freqgrandaverage(cfg, freq_all{:});
%     clear freq_all
%     switch sel_bl
%         case 'z'
%         save (fullfile(path_in,strcat('GA_',sel_freq_band,'_',cond{c},'.mat')),'freq');
%         case 'nobl'
%         save (fullfile(path_in,strcat('GA_',sel_freq_band,'_',cond{c},'nobl.mat')),'freq');
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


%% sensor level stats: 3d cluster stats and figures:
%faces- words main effect contrast

%figure 2 a & figure 2d & cluster tf in supplemental 1A
path_in=fullfile(project_path,'meg_data','freq');
path_out=fullfile(project_path,'meg_data','freq','stats');
mkdir(path_out)

load(fullfile(project_path,'scripts','additional_scripts','neighbours_tri.mat'));
layout_file=(fullfile(project_path,'scripts','additional_scripts','4D148.lay'));
load(fullfile(project_path,'scripts','additional_scripts','red2blue_colormap.mat'));

sub={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
        'vp11';'vp12';'vp14';'vp15';'vp18';...
        'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
        'vp31';'vp32'};

cond={'faces_hit','faces_miss','words_hit','words_miss'};
cd(path_in);

freq_bands={'lf','hf'}
for f=1:numel(freq_bands)
        sel_freq_band=freq_bands{f};

        switch sel_freq_band
            case 'lf'
              sel_frq=[2 30];
              sel_time=[0 1.5];
            case 'hf'
              sel_frq=[40 100];
              sel_time=[0 1.5];
        end
    
    % load all data
    for c=1:numel(cond)
        load(strcat('GA_',sel_freq_band,'_',cond{c},'.mat'))
        GA{c}=freq; 
    end

% define freq (average for face and word)
    data1=GA{1};
    data1.powspctrm=(GA{1}.powspctrm+GA{2}.powspctrm)./2;
    data2=GA{2};
    data2.powspctrm=(GA{3}.powspctrm+GA{4}.powspctrm)./2;
    
    % design matrix    
    subs = size(data1.powspctrm,1);
    design=[1:subs, 1:subs; ones(1,subs), ones(1,subs)*2];

        cfg = [];
        cfg.frequency        = sel_frq;
        cfg.latency     = sel_time;
        cfg.method           = 'montecarlo';
        cfg.statistic        = 'depsamplesT';
        cfg.correctm         = 'cluster';
        cfg.clusteralpha     =0.01;
        cfg.clusterstatistic = 'maxsum';
        cfg.minnbchan        = 2;
        cfg.neighbours = neighbours;
        cfg.tail             = 0;%
        cfg.clustertail             = 0;
        cfg.alpha            = 0.01;
        cfg.numrandomization = 1000;
        cfg.avgoverfreq = 'no';
        cfg.avgovertime = 'no';
        cfg.avgoverchan = 'no';
        cfg.design = design;
        cfg.uvar     = 1;%units of observation
        cfg.ivar     = 2;%indep vars
        cfg.computecritval = 'yes';
        stat = ft_freqstatistics(cfg, data1, data2)
        
  %%%%%%%%%% save stats
  freqstr=strcat(num2str(sel_frq(1)),'to',num2str(sel_frq(2)),'Hz');
  timestr=strcat(num2str(sel_time(1).*1000),'to',num2str(sel_time(2).*1000),'ms');

  save(fullfile(path_out,strcat('3dsensorstats_freq',freqstr,'time',timestr,'.mat')),'stat')
end

%% plot results of stats

path_stats=fullfile(project_path,'meg_data','freq','stats');
path_figs=fullfile(project_path,'figures');
mkdir(path_figs)

freq_bands={'lf','hf'}
alpha_level     =0.05;

load(fullfile(project_path,'scripts','additional_scripts','neighbours_tri.mat'));
layout_file=(fullfile(project_path,'scripts','additional_scripts','4D148.lay'));
load(fullfile(project_path,'scripts','additional_scripts','red2blue_colormap.mat'));

for f=1:numel(freq_bands)
sel_freq_band=freq_bands{f};
        switch sel_freq_band
            case 'lf'
              sel_frq=[2 30];
              sel_time=[0 1.5];
              fig_number='2Aplussupplement1A';
              
            case 'hf'
              sel_frq=[40 100];
              sel_time=[0 1.5];
              fig_number='2Dplussupplement1A';
        end 
  freqstr=strcat(num2str(sel_frq(1)),'to',num2str(sel_frq(2)),'Hz');
  timestr=strcat(num2str(sel_time(1).*1000),'to',num2str(sel_time(2).*1000),'ms');
  file_name=strcat('3dsensorstats_freq',freqstr,'time',timestr);
  load(fullfile(path_stats,strcat(file_name,'.mat')))

  %check for sig clusterst
    pos_ind=[];    
    if isfield(stat,'posclusters') 
        if ~isempty(stat.posclusters)
        if stat.posclusters(1).prob<alpha_level
        pos_ind=find([stat.posclusters(:).prob]<alpha_level);
        end
        end
    end 
    
    neg_ind=[];
    if isfield(stat,'negclusters')
        if ~isempty(stat.negclusters)
        if stat.negclusters(1).prob<alpha_level
        neg_ind=find([stat.negclusters(:).prob]<alpha_level);
        else 
        end
        end
    end 
    
    
   % dummy data for topoplots
data2plot.dimord='chan_time';
data2plot.time=[1, 2];
data2plot.label=stat.label;
 
 % plot cluster stat results
    if ~isempty(neg_ind)
        % gather data for tf plot
        cluster=zeros(size(stat.negclusterslabelmat));
        for n=1:numel(neg_ind)
        cluster_tmp=(stat.negclusterslabelmat==neg_ind(n));
        cluster=cluster+cluster_tmp;
        end
        
        fig= figure % cluster stat tf + topo, tf tstat of sig elecs, text stats
        % cluster TFR
        ax1 = subplot (2,2,1)
        imagesc(stat.time,stat.freq(1:end), squeeze(nansum(cluster(:,1:end,:).*stat.stat(:,1:end,:))).*-1)
        set(gca,'YDir','normal')
        colormap(ax1,'hot')
        colorbar
        title(strcat('negcluster:',strrep(file_name,'_',' ')))

        % cluster topo
        ax2= subplot(2,2,2)
        topo=squeeze(nansum(nansum(cluster.*stat.stat,2),3)).*-1;
        data2plot.avg=repmat(topo,1,2);
        cfg=[];
        cfg.parameter          = 'avg';
        cfg.layout = layout_file;
        cfg.interactive='no';
        cfg.gridscale          =300;
        cfg.style  ='straight';
        cfg.markersize    =6; 
        ft_topoplotER(cfg, data2plot)
        colormap(ax2,'hot')
        colorbar

        ax3= subplot(2,2,3)
        sig_elecs=find(topo);
        mean_tstat=squeeze(nanmean(stat.stat(sig_elecs,:,:)));
        imagesc(stat.time,stat.freq(1:end),mean_tstat(1:end,:),[-3 3])
        set(gca,'YDir','normal')
        colormap(ax3,red2blue)
        colorbar
        title('mean T-value sig elec')

        subplot(2,2,4)
        text(0,0.9,strcat('neg cluster significant'))
        text(0,0.8,strcat('clusteralpha=',num2str(stat.cfg.clusteralpha),' alpha=',num2str(stat.cfg.alpha)))
        for l=1:numel(neg_ind)
        text(0, 0.7-(l*0.1), strcat('pcluster=', num2str(stat.negclusters(l).prob),' T-sum=', num2str(stat.negclusters(l).clusterstat)))
        end
        axis off
        savefig(fig,fullfile(path_figs,strcat('neg_cluster',fig_number,'.fig')))
        close all
        end 

  if ~isempty(pos_ind)
        % gather data for tf plot
        cluster=zeros(size(stat.posclusterslabelmat));
        for n=1:numel(pos_ind)
        cluster_tmp=(stat.posclusterslabelmat==pos_ind(n));
        cluster=cluster+cluster_tmp;
        end

        fig= figure % cluster stat tf + topo, tf tstat of sig elecs, text stats
        % cluster TFR
        ax1 = subplot (2,2,1)
        imagesc(stat.time,stat.freq(1:end), squeeze(nansum(cluster(:,1:end,:).*stat.stat(:,1:end,:))))
        set(gca,'YDir','normal')
        colormap(ax1,'hot')
        colorbar
        title(strcat('poscluster:',strrep(file_name,'_',' ')))

        % cluster topo
        ax2= subplot(2,2,2)
        topo=squeeze(nansum(nansum(cluster.*stat.stat,2),3));
        data2plot.avg=repmat(topo,1,2);
        cfg=[];
        cfg.parameter          = 'avg';
        cfg.layout = layout_file;
        cfg.interactive='no';
        cfg.gridscale          =300;
        cfg.style  ='straight';
        cfg.markersize    =6; 
        ft_topoplotER(cfg, data2plot)
        colormap(ax2,'hot')
        colorbar

        ax3= subplot(2,2,3)
        sig_elecs=find(topo);
        mean_tstat=squeeze(nanmean(stat.stat(sig_elecs,:,:)));
        imagesc(stat.time,stat.freq(1:end),mean_tstat(1:end,:),[-3 3])
        set(gca,'YDir','normal')
        colormap(ax3,red2blue)
        colorbar
        title('mean T-value sig elec')

        subplot(2,2,4)
        text(0,0.9,strcat('pos cluster significant'))
        text(0,0.8,strcat('clusteralpha=',num2str(stat.cfg.clusteralpha),' alpha=',num2str(stat.cfg.alpha)))
        for l=1:numel(pos_ind)
        text(0, 0.7-(l*0.1), strcat('pcluster=', num2str(stat.posclusters(l).prob),' T-sum=', num2str(stat.posclusters(l).clusterstat)))
        end
        axis off
        savefig(fig,fullfile(path_figs,strcat('pos_cluster',fig_number,'.fig')))
        close all
  end  
end



%% base line corrected tfrs (supplemental figure 1 b&c)
path_in=fullfile(project_path,'meg_data','freq');
path_figs=fullfile(project_path,'figures');

layout_file=(fullfile(project_path,'scripts','additional_scripts','4D148.lay'));

% load lf stat & get cluster definition
load(fullfile(path_in,'stats',strcat('3dsensorstats_freq2to30Hztime0to1500ms.mat')))
cluster_elecs{1}=sum(sum(stat.negclusterslabelmat==1,2),3)~=0;
cluster_elecs{2}=sum(sum(stat.posclusterslabelmat==1,2),3)~=0;

cluster_def={'posterior','left frontal'};
cond={'faces','words'};
for c=1:numel(cond)
    sel_cond=cond{c};
load(fullfile(path_in,strcat('GA_lf_',sel_cond,'_hitnobl.mat')))
data1=freq;
load(fullfile(path_in,strcat('GA_lf_',sel_cond,'_missnobl.mat')))
data2=freq;
data=data1;
data.powspctrm=(data1.powspctrm+data2.powspctrm)./2;

fig=figure
for e=1:numel(cluster_elecs)  
subplot(2,2,2+e)
cfg=[];
    cfg.interactive='yes';
    cfg.layout = layout_file;
    cfg.baseline       = [-0.6 -0.1];
    cfg.baselinetype   = 'relchange';
    cfg.xlim=[-0.5 1.5];
    cfg.ylim=[2.5 30];
    cfg.zlim=[-0.5 0.5];
    cfg.channel=data.label(cluster_elecs{e});
    cfg.interactive='no';
ft_singleplotTFR(cfg,data);
end

load(fullfile(path_in,strcat('GA_hf_',sel_cond,'_hitnobl.mat')))
data1=freq;
load(fullfile(path_in,strcat('GA_hf_',sel_cond,'_missnobl.mat')))
data2=freq;

data=data1;
data.powspctrm=(data1.powspctrm+data2.powspctrm)./2;

for e=1:numel(cluster_elecs)  
subplot(2,2,e)
cfg=[];
    cfg.interactive='yes';
    cfg.layout = layout_file;
    cfg.baseline       = [-0.6 -0.1];
    cfg.baselinetype   = 'relchange';
    cfg.xlim=[-0.5 1.5];
    cfg.ylim=[35 100];
    cfg.zlim=[-0.1 0.1];
    cfg.channel=data.label(cluster_elecs{e});
    cfg.title=strcat(sel_cond,' relative change to baseline', cluster_def{e}, 'cluster');
    cfg.interactive='no';
    ft_singleplotTFR(cfg,data);

end
savefig(fig,fullfile(path_figs,strcat('supplementalfig2bc',sel_cond,'.fig')))
close all
end