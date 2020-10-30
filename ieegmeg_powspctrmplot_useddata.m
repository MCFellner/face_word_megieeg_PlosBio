project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%% get powspctrm for used source data
%% get data in source rois
%%%%%%%%%%%%% data definition

% 
vp={'vp01';'vp02';'vp03';'vp05';'vp06';'vp07';'vp09';...
       'vp11';'vp12';'vp14';'vp15';'vp18';...
       'vp22';'vp23';'vp24';'vp27';'vp29';'vp30';...
       'vp31';'vp32'};
% 
path_data=fullfile(project_path,'meg_data','source');
path_fig=fullfile(project_path,'figures');
condition={'hit_words','miss_words','hit_faces','miss_faces'};



% %%%%%%%%%%%%%meg cluster stat definition
 path_stats=fullfile(project_path,'meg_data','source','stats');
% 
% load MEG w/f rois (sourcestats...)
all_freq={'theta','alphabeta','gamma'};
freq_def=[2,5;8 20;50 90];
time_def=[1, 1.5;0.3, 1.5;0.3 1];

all_con={'main_sme'};
% 

for con=1
sel_con=all_con{con};
fig=figure

for f=1:3;

sel_freq_band=all_freq{f};
foi=freq_def(f,:);
toi=time_def(f,:);
effect=all_con{1};

 freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
  timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');
  filename=fullfile(path_stats,strcat('sourcestats_freq',effect,freqstr,'time',timestr,'.mat'));
load(filename);


        
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


template=load(fullfile(project_path,'scripts','additional_scripts','standard_sourcemodel3d10mm'));

for n=1:numel(vp)
 sel_sub=vp{n};
       
   for c=1:numel(condition)
     load(fullfile(path_data,strcat(sel_sub,'beamed_TF_lf', condition{c})));
    freq_lf=freq_cond;
clear freq_cond
     load(fullfile(path_data,strcat(sel_sub,'beamed_TF_hf', condition{c})));
freq_hf=freq_cond;
clear freq_cond
% match time hf to lf
sel_t1=nearest(freq_hf.time,freq_lf.time(1));
sel_t2=nearest(freq_hf.time,freq_lf.time(end));

freq_hf.time=freq_hf.time(sel_t1:sel_t2);
freq_hf.powspctrm=freq_hf.powspctrm(:,:,sel_t1:sel_t2);
% merge lf & hf
freq_cond=freq_lf;
freq_cond.freq=[freq_lf.freq,freq_hf.freq];
freq_cond.powspctrm=cat(2,freq_lf.powspctrm,freq_hf.powspctrm);
     z_pow{c}(n,:,:,:)=(freq_cond.powspctrm);
   end
end
load (fullfile(path_data,strcat(sel_sub,'template_source'))) %load a template source, dummy for later stats and plotting

  % match indices of stat with pow
sourceAll.pos = template.sourcemodel.pos;
sourceAll.dim = template.sourcemodel.dim;
insidepos=find(sourceAll.inside);

pos_cluster_def=pos_cluster_def(insidepos);
neg_cluster_def=neg_cluster_def(insidepos); 



% %%%%%%%%%% pos cluster
% % plot
 t1=nearest(freq_cond.time,toi(1));
 t2=nearest(freq_cond.time,toi(2));
 f1=nearest(freq_cond.time,foi(1));
f2=nearest(freq_cond.time,foi(2));
 
% 

switch sel_freq_band
    case 'gamma'
 if sum (pos_cluster_def>0)
    for c=1:numel(condition)  
     pos_mean_zpow{c}=squeeze(nanmean(nanmean(nanmean(z_pow{c}(:,pos_cluster_def,:,t1:t2),1),2),4));
     pos_mean_meanzpow{c}=squeeze(nanmean(nanmean(nanmean(nanmean(z_pow{c}(:,pos_cluster_def,f1:f2,t1:t2),1),2),3),4));

     pos_all_zpow{c}=squeeze(nanmean(nanmean(z_pow{c}(:,pos_cluster_def,:,t1:t2),2),4));
     pos_all_meanzpow{c}=squeeze(nanmean(nanmean(nanmean(z_pow{c}(:,pos_cluster_def,f1:f2,t1:t2),2),3),4));

    end
 end

% fig pos cluster

 if sum (pos_cluster_def>0)

% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
subplot(1,3,f)
for c=1:numel(condition)
plot((freq_cond.freq),(pos_mean_zpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on

% plot slope lines for the different fits (only for fitted freqs)
end
legend(condition)

switch sel_con
    case 'main_sme'
      cond1=((pos_all_zpow{1}+pos_all_zpow{3}).*0.5);
      cond2=((pos_all_zpow{2}+pos_all_zpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot(((freq_cond.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

xlabel('freq')
ylabel('power')
 end



 otherwise
 if sum (neg_cluster_def>0)
for c=1:numel(condition)
     
     neg_mean_zpow{c}=squeeze(nanmean(nanmean(nanmean(z_pow{c}(:,neg_cluster_def,:,t1:t2),1),2),4));
     neg_mean_meanzpow{c}=squeeze(nanmean(nanmean(nanmean(nanmean(z_pow{c}(:,neg_cluster_def,f1:f2,t1:t2),1),2),3),4));

     neg_all_zpow{c}=squeeze(nanmean(nanmean(z_pow{c}(:,neg_cluster_def,:,t1:t2),2),4));
     neg_all_meanzpow{c}=squeeze(nanmean(nanmean(nanmean(z_pow{c}(:,neg_cluster_def,f1:f2,t1:t2),2),3),4));

end
 end
 
if sum (neg_cluster_def>0)

%%%%%%%%%%%%%% fig neg cluster
% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
subplot(1,3,f)

for c=1:numel(condition)
plot((freq_cond.freq),(neg_mean_zpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on
end

legend(condition)
switch sel_con
    case 'main_sme'
      cond1=((neg_all_zpow{1}+neg_all_zpow{3}).*0.5);
      cond2=((neg_all_zpow{2}+neg_all_zpow{4}).*0.5);
    otherwise        
        display ('unknown contrast')
end
a=gca;
plot_y=a.YLim(2);
[h,p]=ttest(cond1,cond2)
plot(((freq_cond.freq(h==1))),repmat(plot_y,sum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq_cond.freq(h_corr==1))),repmat(plot_y,sum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')
end
end
end
end
savefig(fig,fullfile(path_fig,strcat('suppfig5_MEG','org_z_data')))


%%%%%%%%%%%%iEEG
%%
%% get powspctrm for used ieeg data
%% combine lf and hf files
  path_in=fullfile(project_path,'ieeg_data','freq');
path_stats=fullfile(path_in,'stats');
%  
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};

cond={'words','faces'};
mem={'hit','miss'};
load(fullfile(path_stats,'allelecstatcondalphabeta8to20Hz300to1500ms.mat'))
condition={'wordshit','wordsmiss','faceshit','facesmiss'};
for c=1:numel(cond)
 for n=1:numel(pat)
nan_def=find(isnan(allstat{n}.h));  
 sel_elec=allstat{n}.elecs;
sel_elec(nan_def)=[];
%[~,~,loc_elec]=intersect(sel_elec,data.label);

 % load lf
load(fullfile(path_in,strcat(pat{n},'_lf_',cond{c})))
      cfg=[];
      cfg.channel=sel_elec;
      freq = ft_selectdata(cfg, freq)
      freq=rmfield(freq,'cfg');
      freq_lf=freq;

 % load hf
load(fullfile(path_in,strcat(pat{n},'_hf_',cond{c})))
      cfg=[];
      cfg.channel=sel_elec;
      freq = ft_selectdata(cfg, freq)
      freq=rmfield(freq,'cfg');
      freq_hf=freq;
      clear freq
      
      % match time hf to lf
    sel_t1=nearest(freq_lf.time,freq_hf.time(1));
    sel_t2=nearest(freq_lf.time,freq_hf.time(end));

    freq_lf.time=freq_lf.time(sel_t1:sel_t2);
    freq_lf.powspctrm=freq_lf.powspctrm(:,:,:,sel_t1:sel_t2);
    
     % match freq hf to lf
    sel_f1=nearest(freq_lf.freq,freq_lf.freq(1));
    sel_f2=nearest(freq_lf.freq,freq_hf.freq(1));
    freq_lf.freq=freq_lf.freq(sel_f1:sel_f2);
    freq_lf.powspctrm=freq_lf.powspctrm(:,:,sel_f1:sel_f2,:);
    
    % merge lf & hf
freq=freq_lf;
freq.freq=[freq_lf.freq,freq_hf.freq];
freq.powspctrm=cat(3,freq_lf.powspctrm,freq_hf.powspctrm);
       
 freq_tmp=freq;
           for m=1:size(mem,2)
             if m==1
               trials=find(freq_tmp.trialinfo(:,4)>=1);
              else
               trials=find(freq_tmp.trialinfo(:,4)==0);
              end
               freq=freq_tmp;
                freq.powspctrm=squeeze(nanmean(freq_tmp.powspctrm(trials,:,:,:),1));              
                freq.dimord='chan_freq_time';
                freq.trialinfo=freq_tmp.trialinfo(trials,:);
            all_freq{n,(c*(c-1)+m)}=freq;                   
         end

end
 end
 save(strcat(path_in,'GA_allfreqcombi'), 'all_freq','condition')


%%
% get condition average for electrodes based on allstat
path_figs=fullfile(project_path,'figures');

path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(path_in,'stats');
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};


% load MEG w/f rois (sourcestats...)
% define allstat file to load
freqs={'theta','alphabeta','gamma'};
freq_def=[2,5;8 20;50 90];
time_def=[1, 1.5;0.3, 1.5;0.3 1];
all_con={'mem'};

condition={'words_hit','words_miss','faces_hit','faces_miss'};

for con=1;
sel_con=all_con{con};
fig=figure

for f=1:3;
sel_freq_band=freqs{f};
foi=freq_def(f,:);
toi=time_def(f,:);

freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms'); 
load(fullfile(path_out,strcat('allelecstat', sel_con,sel_freq_band,freqstr,timestr)));

% load allfreq combi (all patients/cond/freq combined)
load(strcat(path_in,'GA_allfreqcombi'))

for n=1:numel(pat)
% define roi electrodes,  define pos/neg cluster
pos_cluster_def_all{n}=allstat{n}.h==1&allstat{n}.f>0;
neg_cluster_def_all{n}=allstat{n}.h==1&allstat{n}.f<0;
  nan_def=find(isnan(allstat{n}.h));  
% delete all nan channels  
  pos_cluster_def_all{n}(nan_def)=[];
  neg_cluster_def_all{n}(nan_def)=[];
  
for c=1:numel(condition)


freq_all{n,c}.org_pow=all_freq{n,c}.powspctrm; 
freq.freq=all_freq{1,1}.freq;
freq.time=all_freq{1,1}.time;

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
   
    tmp_org_pow{n}=freq_all{n,c}.org_pow;

    end
   
      org_pow{c}= vertcat(tmp_org_pow{:});
end

% delete all nan channels
pos_cluster_def=[pos_cluster_def_all{:}];
neg_cluster_def=[neg_cluster_def_all{:}];


if sum (pos_cluster_def)>0   
for c=1:numel(condition)
pos_mean_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(pos_cluster_def,:,t1:t2),1),3));
pos_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(pos_cluster_def,f1:f2,t1:t2),1),2),3));

pos_all_orgpow{c}=squeeze(nanmean(org_pow{c}(pos_cluster_def,:,t1:t2),3));
pos_all_meanorgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(pos_cluster_def,f1:f2,t1:t2),2),3));
end
end

if sum (neg_cluster_def)>0
for c=1:numel(condition)
neg_mean_orgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(neg_cluster_def,:,t1:t2),1),3));
neg_mean_meanorgpow{c}=squeeze(nanmean(nanmean(nanmean(org_pow{c}(neg_cluster_def,f1:f2,t1:t2),1),2),3));

neg_all_orgpow{c}=squeeze(nanmean(org_pow{c}(neg_cluster_def,:,t1:t2),3));
neg_all_meanorgpow{c}=squeeze(nanmean(nanmean(org_pow{c}(neg_cluster_def,f1:f2,t1:t2),2),3));
end

end


% fig pos cluster

color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};
subplot(1,3,f)

switch sel_freq_band
    
    case 'gamma'
for c=1:numel(condition)
plot((freq.freq),(pos_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on
end

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
plot(((freq.freq(h==1))),repmat(plot_y,nansum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot(((freq.freq(h_corr==1))),repmat(plot_y,nansum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

title(strcat('poscluster:','allstat_',sel_con,sel_freq_band, num2str(foi(1)),'to', num2str(foi(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))
legend(condition)

xlabel('freq')
ylabel('power')

    otherwise
        
%%%%%%%%%%%%%% fig neg cluster
% default colors
color_def=  [0  0.4470 0.7410; 0.8500 0.3250  0.0980;  0.9290  0.6940 0.1250;  0.4940 0.1840 0.5560];
line_def={'--',':','-.'};


subplot(1,3,f)
for c=1:numel(condition)
plot((freq.freq),(neg_mean_orgpow{c}),'LineWidth',3,'Color', color_def(c,:))
hold on

end
xlabel('freq')
ylabel('power')

legend(condition)
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
plot(((freq.freq(h==1))),repmat(plot_y,nansum(h),1),'LineStyle','none','Marker','o','MarkerEdgeColor','k')
[pthr,pcor,padj] = fdr(p);
h_corr=p<pthr;
plot((freq.freq(h_corr==1)),repmat(plot_y,nansum(h_corr),1),'LineStyle','none','Marker','o','MarkerEdgeColor','r')

title(strcat('negcluster:','allstat_',sel_con,sel_freq_band, num2str(foi(1)),'to', num2str(foi(2)),'time',num2str(toi(1).*1000),'to', num2str(toi(2).*1000)))
end
end
end

savefig(fig,fullfile(path_figs,strcat('suppfig5_iEEG','org_z_data')))
