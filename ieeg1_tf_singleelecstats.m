project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%% freq analysis

% the files generated here are already part of the provided data
% path_in=fullfile(project_path,'ieeg_data');
% path_out=fullfile(path_in,'freq');
% mkdir (path_out);
% 
% subs={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
% 
% cond={'words','faces'};
% freq_bands={'lf','hf'}
% for f=1:numel(freq_bands)
%     sel_freq_band=freq_bands{f};
%     for c=1:numel(cond)
%         for n=1:numel(subs)
%             sub=subs{n};
%             load(fullfile(path_in,strcat(sub,'_',cond{c})));
%              elecinfo=data.elecinfo;
% 
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
%          end     
% 
%         %select response free/edge artifact free interval for z trans       
%               cfg=[];
%               cfg.time=[-0.5 1.5];
%               freq=z_trans_TF_seltime(cfg,freq);
%               freq.elecinfo=elecinfo;
%               
%         %smooth single trials
%              cfg=[]; 
%              cfg.fwhm_t=0.2;
%              switch sel_freq_band
%                  case 'lf'
%              cfg.fwhm_f=2;             
%                  case 'hf'
%              cfg.fwhm_f=10;
%              end
%              freq= smooth_TF(cfg,freq)
% 
% 
%         save(fullfile(path_out,strcat(sub,'_',sel_freq_band,'_',cond{c},'.mat')),'freq')
%      end
%     end
% end

%% single electrode stats
path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(path_in,'stats');
path_figs=fullfile(project_path,'figures');

mkdir (path_out);
% all patients with word and face session
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
freq_bands={'theta','alphabeta','gamma'};

alpha_level=0.05;

min_t=10; % minimum trials allowed per condition
ecount=0;
allcount=0;
for f=1:numel(freq_bands)
    % define depending on freq band
    % tf for finding electrodes
        sel_freq_band=freq_bands{f};
        switch sel_freq_band
            case 'alphabeta'
                foi        = [8 20];
                toi     = [0.3 1.5];
                effects={'mem','cond'};
                fig_numbers={'fig4d_elecs','fig2c_elecs'};
                freq_def='lf';
                sme_direction=-1;
            case 'theta'
                foi        = [2 5];
                toi     = [1 1.5];
                effects={'mem'};
                fig_numbers={'fig4b_elecs'};
                freq_def='lf';
                sme_direction=-1;

            case 'gamma'
                foi        = [50 90];
                toi     = [0.3 1];
                effects={'mem','cond'};
                fig_numbers={'fig4e_elecs','fig2f_elecs'};
                freq_def='hf';
                sme_direction=1;

        end  
    
for eff=1:numel(effects)
    effect=effects{eff}
for n=1:numel(pat)
%load data
load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_faces')))
data1=freq;
load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_words')))
data2=freq;
clear freq

% get toi and foi 
f1=nearest(data1.freq,foi(1));
f2=nearest(data1.freq,foi(2));
t1=nearest(data1.time,toi(1));
t2=nearest(data1.time,toi(2));
% define trials
hitind1=find(data1.trialinfo(:,4)==1);
hitind2=find(data2.trialinfo(:,4)==1);
missind1=find(data1.trialinfo(:,4)==0);
missind2=find(data2.trialinfo(:,4)==0);
all_freq_avg=[];
    for e=1:numel(data1.label)
    freq1=squeeze(nanmean(nanmean(data1.powspctrm(hitind1,e,f1:f2,t1:t2),3),4)); 
    freq2=squeeze(nanmean(nanmean(data1.powspctrm(missind1,e,f1:f2,t1:t2),3),4));   
    freq3=squeeze(nanmean(nanmean(data2.powspctrm(hitind2,e,f1:f2,t1:t2),3),4)); 
    freq4=squeeze(nanmean(nanmean(data2.powspctrm(missind2,e,f1:f2,t1:t2),3),4));
    all_freq_avg=[all_freq_avg;nanmean(freq1),nanmean(freq2),nanmean(freq3),nanmean(freq4)];
    % check for minimum trial number in each condition, if not enough
    % set values NaN
        if (sum(~isnan(freq1)))<min_t|(sum(~isnan(freq2)))<min_t|(sum(~isnan(freq3)))<min_t|(sum(~isnan(freq4)))<min_t
        h(e)=NaN;
        p(e)=NaN;
        t(e)=NaN;
        else           
        freq=[freq1;freq2;freq3;freq4];
        mem=[repmat({'hit'},size(hitind1));repmat({'miss'},size(missind1));repmat({'hit'},size(hitind2));repmat({'miss'},size(missind2))];
        cond=[repmat({'word'},size([hitind1;missind1]));repmat({'face'},size([hitind2;missind2]))];
            if strmatch(effect,'cond')
            [p_tmp,tbl,stats] = anovan(freq,{mem cond},'model','interaction','varnames',{'mem','cond'},'display','off');
            h(e)=double(p_tmp(2)<alpha_level);
            p(e)=p_tmp(2);               
            % get direction of difference for later plotting
            direction=sign(((nanmean(freq1)+nanmean(freq2)).*0.5)-((nanmean(freq3)+nanmean(freq4)).*0.5));
            f(e)=tbl{3,6}.*direction;
            elseif strmatch(effect,'mem')
            [p_tmp,tbl,stats] = anovan(freq,{mem cond},'model','interaction','varnames',{'mem','cond'},'display','off');
            h(e)=double(p_tmp(1)<alpha_level);
            p(e)=p_tmp(1);
            % get direction of difference for later plotting
            direction=sign(((nanmean(freq1)+nanmean(freq3)).*0.5)-((nanmean(freq2)+nanmean(freq4)).*0.5));           
            f(e)=tbl{2,6}.*direction;              
            end
        end
        
    end
% get position of electrods    
allstat{n}.elecpos=data1.elecinfo.elecpos_bipolar;      
allstat{n}.h=h;
allstat{n}.p=p;
allstat{n}.f=f;
allstat{n}.elecs=data1.label;
allstat{n}.all_freq_avg=all_freq_avg;

ecount=ecount+sum(h==1);
allcount=allcount+sum(~isnan(allstat{n}.h));
clear h p f freq1 freq2 freq3 freq4 hitind1 missind1 hitind2 missind2
end
% save allstat   
freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms'); 
save(fullfile(path_out,strcat('allelecstat', effect,sel_freq_band,freqstr,timestr)),'allstat')

% plot electrodes (note these figures are slightly different from in the
% manuscript (manuscript figure are plotted with caret)) here using new
% fieldtrip ecog toolboxextension
allstatplot_ft2019(allstat,'no_cluster',path_figs,fig_numbers{eff},project_path,fieldtrip_path);


% for reproducing caret plots use the here exported foci/ foci color files
% and plot foci on spm surface
path_foci=fullfile(path_out,'foci');
mkdir(path_foci)
file_name= strcat(effect,sel_freq_band,freqstr,timestr);
allstat2caretfoci(allstat,'no_cluster',pat,path_foci,file_name);

% get bargraph (figure 4)
switch effect
    case 'mem'
    for n=1:numel(pat)
    % check for each pat for significant electrodes
    all_h=allstat{n}.h ;
    all_h(isnan(all_h))=0;
     sel_elec=all_h & (sign(allstat{n}.f)==sme_direction );
     
     if nansum(sel_elec)==0
         avg_pow(n,:)=nan(1,4);
     else
     avg_pow(n,:)=nanmean(allstat{n}.all_freq_avg(sel_elec,:),1);
     end
    end
    

% bar plots
% reorganize avg_pow
% remove pat with no elec
avg_pow(isnan(avg_pow(:,1)),:)=[];
% change condition order
conditions={'word hit','word miss','face hit','face miss'}
avg_pow=avg_pow(:,[3,4,1,2]);
fig=figure
  subplot(1,2,1)
  hold on
  bar([nanmean(avg_pow)])
  ylim([-0.4 0.4])  
  scatter(reshape(repmat(1:4,size(avg_pow,1),1),[],1),reshape(avg_pow,[],1),'k');
  ax=gca;
  ax.XTick=(1:4);
  ax.XTickLabel=conditions;
  title(strcat(effect,':',sel_freq_band,' ',freqstr,' ',timestr))

    % ttest for interaction
   [h,p,~,tstat]=ttest(avg_pow(:,1)-avg_pow(:,2),avg_pow(:,3)-avg_pow(:,4)); 
   subplot(1,2,2)
        text(0,0.9,strcat('interaction'))
        text(0,0.8,strcat('p=',num2str(p),' T(',num2str(tstat.df),')=',num2str(tstat.tstat)))
        axis off
   savefig(fig,fullfile(path_figs,strcat(fig_numbers{eff},'_barplot'))); 
   close all
end
end
end



%% plot bargraphs of figure 3
path_in=fullfile(project_path,'ieeg_data','freq');
path_stats=fullfile(project_path,'ieeg_data','freq','stats');
path_figs=fullfile(project_path,'figures');



% load allstat cond alpha/beta 
load(fullfile(path_stats,strcat('allelecstatcondalphabeta8to20Hz300to1500ms.mat')))
fig_numbers={'fig3B','fig3D'};

% get gamma power in alpha/beta electrodes for each patient
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
foi=[50 90];
toi=[0.3 1];

for n=1:numel(pat)
load(fullfile(path_in,strcat(pat{n},'_hf_words')))
data1=freq;
load(fullfile(path_in,strcat(pat{n},'_hf_faces')))
data2=freq;
clear freq
h=allstat{n}.h.*sign(allstat{n}.f);

pos_elecs{n}.label=allstat{n}.elecs(find(h==1));
neg_elecs{n}.label=allstat{n}.elecs(find(h==-1));

hitind1=find(data1.trialinfo(:,4)==1);
hitind2=find(data2.trialinfo(:,4)==1);
missind1=find(data1.trialinfo(:,4)==0);
missind2=find(data2.trialinfo(:,4)==0);

[l,ind_pos]=intersect(data1.label,pos_elecs{n}.label);
[l,ind_neg]=intersect(data1.label,neg_elecs{n}.label);

pos_elecs{n}.word_hit=squeeze(nanmean(data1.powspctrm(hitind1,ind_pos,:,:),1));
pos_elecs{n}.word_miss=squeeze(nanmean(data1.powspctrm(missind1,ind_pos,:,:),1));
pos_elecs{n}.face_hit=squeeze(nanmean(data2.powspctrm(hitind2,ind_pos,:,:),1));
pos_elecs{n}.face_miss=squeeze(nanmean(data2.powspctrm(missind2,ind_pos,:,:),1));

neg_elecs{n}.word_hit=squeeze(nanmean(data1.powspctrm(hitind1,ind_neg,:,:),1));
neg_elecs{n}.word_miss=squeeze(nanmean(data1.powspctrm(missind1,ind_neg,:,:),1));
neg_elecs{n}.face_hit=squeeze(nanmean(data2.powspctrm(hitind2,ind_neg,:,:),1));
neg_elecs{n}.face_miss=squeeze(nanmean(data2.powspctrm(missind2,ind_neg,:,:),1));

clear ind_pos ind_neg
end


for n=1:numel(allstat)
pos_count(n)=numel(pos_elecs{n}.label);
neg_count(n)=numel(neg_elecs{n}.label);
end


contrasts={'neg','pos'};
for sel_dir=1:numel(contrasts);
totest=contrasts{sel_dir};

    switch totest
        case 'pos'
        pat_ind=find(pos_count);
        data=pos_elecs;
        case 'neg'
        pat_ind=find(neg_count);
        data=neg_elecs;    
        otherwise
    end
    
  % prepare data for groupstats (avg elecs in pat, powspctrm(pat,1,freq,time)
    for p=1:numel(pat_ind)
        
        sel_pat=pat_ind(p);
        if numel(data{sel_pat}.label)>1
        pow1(p,1,:,:)=squeeze(nanmean(data{sel_pat}.word_hit));
        pow2(p,1,:,:)=squeeze(nanmean(data{sel_pat}.word_miss));
        pow3(p,1,:,:)=squeeze(nanmean(data{sel_pat}.face_hit));
        pow4(p,1,:,:)=squeeze(nanmean(data{sel_pat}.face_miss));        
        else
        pow1(p,1,:,:)=data{sel_pat}.word_hit; 
        pow2(p,1,:,:)=data{sel_pat}.word_miss; 
        pow3(p,1,:,:)=data{sel_pat}.face_hit; 
        pow4(p,1,:,:)=data{sel_pat}.face_miss;         
        end
    end
    
   f1=nearest(data1.freq,foi(1));
   f2=nearest(data1.freq,foi(2));
   t1=nearest(data1.time,toi(1));
   t2=nearest(data1.time,toi(2));
    
  cond1=squeeze(nanmean(nanmean(pow1(:,:,f1:f2,t1:t2),3),4));
  cond2=squeeze(nanmean(nanmean(pow2(:,:,f1:f2,t1:t2),3),4));
  cond3=squeeze(nanmean(nanmean(pow3(:,:,f1:f2,t1:t2),3),4));
  cond4=squeeze(nanmean(nanmean(pow4(:,:,f1:f2,t1:t2),3),4));  
  
  % avg word and face
  avg_pow=[(cond1+cond2).*0.5,(cond3+cond4).*0.5];
    conditions={'word','face'}

fig=figure
  subplot(1,3,1)
  hold on
  bar([nanmean(avg_pow)])
  ylim([-0.1 0.2])  
  scatter(reshape(repmat(1:2,size(avg_pow,1),1),[],1),reshape(avg_pow,[],1),'k');
  ax=gca;
  ax.XTick=(1:2);
  ax.XTickLabel=conditions;
  title(strcat('gamma in ',totest,' alphabeta electrodes'))

    % ttest for interaction
   [h,p,~,tstat]=ttest(avg_pow(:,1),avg_pow(:,2)); 
   subplot(1,3,2)
        text(0,0.9,strcat('main cond effect'))
        text(0,0.8,strcat('p=',num2str(p),' T(',num2str(tstat.df),')=',num2str(tstat.tstat)))
        axis off
        
    subplot(1,3,3)
    % plot tf 
    [~,~,~,tstat]=ttest((pow1+pow2).*0.5,(pow3+pow4).*0.5); 

    toi_plot=[0,1.5];
    tp1=nearest(data1.time,toi_plot(1));
    tp2=nearest(data1.time,toi_plot(2));
    foi_plot=[30 100];
    fp1=nearest(data1.freq,foi_plot(1));
    fp2=nearest(data1.freq,foi_plot(2));

    imagesc(data1.time(tp1:tp2),data1.freq(fp1:fp2),squeeze(tstat.tstat(:,:,fp1:fp2,tp1:tp2)).*-1,[-4 4]);
    set(gca, 'Ydir','normal')
    colormap('jet')
    colorbar
   savefig(fig,fullfile(path_figs,strcat(fig_numbers{sel_dir}))); 
  
clear pow1 pow2 pow3 pow4 avg_pow cond1 cond2 cond3 cond4 
end

%% plot bargraphs & tf of figure 5: gamma in theta
path_in=fullfile(project_path,'ieeg_data','freq');
path_stats=fullfile(project_path,'ieeg_data','freq','stats');
path_figs=fullfile(project_path,'figures');



% load allstat mem theta 
load(fullfile(path_stats,strcat('allelecstatmemtheta2to5Hz1000to1500ms.mat')))
fig_number='fig5de';

% get gamma power in alpha/beta electrodes for each patient
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
foi=[50 90];
toi1=[0.3 1];
toi2=[1 1.5];

for n=1:numel(pat)
load(fullfile(path_in,strcat(pat{n},'_lf_words')))
lf_data1=freq;
load(fullfile(path_in,strcat(pat{n},'_lf_faces')))
lf_data2=freq;
clear freq
load(fullfile(path_in,strcat(pat{n},'_hf_words')))
hf_data1=freq;
load(fullfile(path_in,strcat(pat{n},'_hf_faces')))
hf_data2=freq;
clear freq

h=allstat{n}.h.*sign(allstat{n}.f);

neg_elecs{n}.label=allstat{n}.elecs(find(h==-1));

hitind1=find(hf_data1.trialinfo(:,4)==1);
hitind2=find(hf_data2.trialinfo(:,4)==1);
missind1=find(hf_data1.trialinfo(:,4)==0);
missind2=find(hf_data2.trialinfo(:,4)==0);

[l,ind_neg]=intersect(hf_data1.label,neg_elecs{n}.label);

hf_neg_elecs{n}.word_hit=squeeze(nanmean(hf_data1.powspctrm(hitind1,ind_neg,:,:),1));
hf_neg_elecs{n}.word_miss=squeeze(nanmean(hf_data1.powspctrm(missind1,ind_neg,:,:),1));
hf_neg_elecs{n}.face_hit=squeeze(nanmean(hf_data2.powspctrm(hitind2,ind_neg,:,:),1));
hf_neg_elecs{n}.face_miss=squeeze(nanmean(hf_data2.powspctrm(missind2,ind_neg,:,:),1));

lf_neg_elecs{n}.word_hit=squeeze(nanmean(lf_data1.powspctrm(hitind1,ind_neg,:,:),1));
lf_neg_elecs{n}.word_miss=squeeze(nanmean(lf_data1.powspctrm(missind1,ind_neg,:,:),1));
lf_neg_elecs{n}.face_hit=squeeze(nanmean(lf_data2.powspctrm(hitind2,ind_neg,:,:),1));
lf_neg_elecs{n}.face_miss=squeeze(nanmean(lf_data2.powspctrm(missind2,ind_neg,:,:),1));
clear ind_pos ind_neg
end


for n=1:numel(allstat)
neg_count(n)=numel(neg_elecs{n}.label);
end


contrasts={'neg'};
totest=contrasts{1};

    switch totest
        case 'neg'
        pat_ind=find(neg_count);
        lf_data=lf_neg_elecs;
        hf_data=hf_neg_elecs; 
        otherwise
    end
    
  % prepare data for groupstats (avg elecs in pat, powspctrm(pat,1,freq,time)
    for p=1:numel(pat_ind)
        
        sel_pat=pat_ind(p);
        if numel(size(hf_data{sel_pat}.word_hit))>2
        hf_pow1(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.word_hit));
        hf_pow2(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.word_miss));
        hf_pow3(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.face_hit));
        hf_pow4(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.face_miss)); 
        lf_pow1(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.word_hit));
        lf_pow2(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.word_miss));
        lf_pow3(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.face_hit));
        lf_pow4(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.face_miss));  
        else
        hf_pow1(p,1,:,:)=hf_data{sel_pat}.word_hit; 
        hf_pow2(p,1,:,:)=hf_data{sel_pat}.word_miss; 
        hf_pow3(p,1,:,:)=hf_data{sel_pat}.face_hit; 
        hf_pow4(p,1,:,:)=hf_data{sel_pat}.face_miss;
        lf_pow1(p,1,:,:)=lf_data{sel_pat}.word_hit; 
        lf_pow2(p,1,:,:)=lf_data{sel_pat}.word_miss; 
        lf_pow3(p,1,:,:)=lf_data{sel_pat}.face_hit; 
        lf_pow4(p,1,:,:)=lf_data{sel_pat}.face_miss;  
        end
    end
    
   f1=nearest(hf_data1.freq,foi(1));
   f2=nearest(hf_data1.freq,foi(2));
   t1=nearest(hf_data1.time,toi1(1));
   t2=nearest(hf_data1.time,toi1(2));
      
   t3=nearest(hf_data1.time,toi2(1));
   t4=nearest(hf_data1.time,toi2(2));
  cond1=squeeze(nanmean(nanmean(hf_pow1(:,:,f1:f2,t1:t2),3),4));
  cond2=squeeze(nanmean(nanmean(hf_pow2(:,:,f1:f2,t1:t2),3),4));
  cond3=squeeze(nanmean(nanmean(hf_pow3(:,:,f1:f2,t1:t2),3),4));
  cond4=squeeze(nanmean(nanmean(hf_pow4(:,:,f1:f2,t1:t2),3),4));  
  
  cond5=squeeze(nanmean(nanmean(hf_pow1(:,:,f1:f2,t3:t4),3),4));
  cond6=squeeze(nanmean(nanmean(hf_pow2(:,:,f1:f2,t3:t4),3),4));
  cond7=squeeze(nanmean(nanmean(hf_pow3(:,:,f1:f2,t3:t4),3),4));
  cond8=squeeze(nanmean(nanmean(hf_pow4(:,:,f1:f2,t3:t4),3),4)); 
  
  % avg word and face
  avg_pow=[(cond1+cond3).*0.5,(cond2+cond4).*0.5,(cond5+cond7).*0.5,(cond6+cond8).*0.5];
    conditions={'rem','forg','rem','forg'}

fig=figure
  subplot(2,2,1)
  hold on
  bar([nanmean(avg_pow)])
  ylim([-0.1 0.2])  
  scatter(reshape(repmat(1:4,size(avg_pow,1),1),[],1),reshape(avg_pow,[],1),'k');
  ax=gca;
  ax.XTick=(1:4);
  ax.XTickLabel=conditions;
  title(strcat('gamma in ',totest,' theta electrodes'))

        
    subplot(2,2,2)
    % plot tf 
    [~,~,~,tstat]=ttest((hf_pow1+hf_pow3).*0.5,(hf_pow2+hf_pow4).*0.5); 

    toi_plot=[0,1.5];
    tp1=nearest(hf_data1.time,toi_plot(1));
    tp2=nearest(hf_data1.time,toi_plot(2));
    foi_plot=[30 100];
    fp1=nearest(hf_data1.freq,foi_plot(1));
    fp2=nearest(hf_data1.freq,foi_plot(2));

    imagesc(hf_data1.time(tp1:tp2),hf_data1.freq(fp1:fp2),squeeze(tstat.tstat(:,:,fp1:fp2,tp1:tp2)),[-4 4]);
    set(gca, 'Ydir','normal')
    colormap('jet')
    colorbar
    
     subplot(2,2,4)
    % plot tf 
    [~,~,~,tstat]=ttest((lf_pow1+lf_pow3).*0.5,(lf_pow2+lf_pow4).*0.5); 

    toi_plot=[0,1.5];
    tp1=nearest(lf_data1.time,toi_plot(1));
    tp2=nearest(lf_data1.time,toi_plot(2));
    foi_plot=[1 30];
    fp1=nearest(lf_data1.freq,foi_plot(1));
    fp2=nearest(lf_data1.freq,foi_plot(2));

    imagesc(lf_data1.time(tp1:tp2),lf_data1.freq(fp1:fp2),squeeze(tstat.tstat(:,:,fp1:fp2,tp1:tp2)),[-4 4]);
    set(gca, 'Ydir','normal')
    colormap('jet')
    colorbar
    
   savefig(fig,fullfile(path_figs,fig_number)); 
  
%% plot bargraphs & tf of figure 5: theta in gamma
path_in=fullfile(project_path,'ieeg_data','freq');
path_stats=fullfile(project_path,'ieeg_data','freq','stats');
path_figs=fullfile(project_path,'figures');



% load allstat mem gamma 
load(fullfile(path_stats,strcat('allelecstatmemgamma50to90Hz300to1000ms.mat')))
fig_number='fig5ab';

% get gamma power in alpha/beta electrodes for each patient
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
foi=[2 5];
toi1=[0.3 1];
toi2=[1 1.5];

for n=1:numel(pat)
load(fullfile(path_in,strcat(pat{n},'_lf_words')))
lf_data1=freq;
load(fullfile(path_in,strcat(pat{n},'_lf_faces')))
lf_data2=freq;
clear freq
load(fullfile(path_in,strcat(pat{n},'_hf_words')))
hf_data1=freq;
load(fullfile(path_in,strcat(pat{n},'_hf_faces')))
hf_data2=freq;
clear freq

h=allstat{n}.h.*sign(allstat{n}.f);

pos_elecs{n}.label=allstat{n}.elecs(find(h==1));

hitind1=find(hf_data1.trialinfo(:,4)==1);
hitind2=find(hf_data2.trialinfo(:,4)==1);
missind1=find(hf_data1.trialinfo(:,4)==0);
missind2=find(hf_data2.trialinfo(:,4)==0);

[l,ind_pos]=intersect(hf_data1.label,pos_elecs{n}.label);

hf_pos_elecs{n}.word_hit=squeeze(nanmean(hf_data1.powspctrm(hitind1,ind_pos,:,:),1));
hf_pos_elecs{n}.word_miss=squeeze(nanmean(hf_data1.powspctrm(missind1,ind_pos,:,:),1));
hf_pos_elecs{n}.face_hit=squeeze(nanmean(hf_data2.powspctrm(hitind2,ind_pos,:,:),1));
hf_pos_elecs{n}.face_miss=squeeze(nanmean(hf_data2.powspctrm(missind2,ind_pos,:,:),1));

lf_pos_elecs{n}.word_hit=squeeze(nanmean(lf_data1.powspctrm(hitind1,ind_pos,:,:),1));
lf_pos_elecs{n}.word_miss=squeeze(nanmean(lf_data1.powspctrm(missind1,ind_pos,:,:),1));
lf_pos_elecs{n}.face_hit=squeeze(nanmean(lf_data2.powspctrm(hitind2,ind_pos,:,:),1));
lf_pos_elecs{n}.face_miss=squeeze(nanmean(lf_data2.powspctrm(missind2,ind_pos,:,:),1));
clear ind_pos ind_pos
end


for n=1:numel(allstat)
pos_count(n)=numel(pos_elecs{n}.label);
end


contrasts={'pos'};
totest=contrasts{1};

    switch totest
        case 'pos'
        pat_ind=find(pos_count);
        lf_data=lf_pos_elecs;
        hf_data=hf_pos_elecs; 
        otherwise
    end
    
  % prepare data for groupstats (avg elecs in pat, powspctrm(pat,1,freq,time)
    for p=1:numel(pat_ind)
        
        sel_pat=pat_ind(p);
        if numel(size(hf_data{sel_pat}.word_hit))>2
        hf_pow1(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.word_hit));
        hf_pow2(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.word_miss));
        hf_pow3(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.face_hit));
        hf_pow4(p,1,:,:)=squeeze(nanmean(hf_data{sel_pat}.face_miss)); 
        lf_pow1(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.word_hit));
        lf_pow2(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.word_miss));
        lf_pow3(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.face_hit));
        lf_pow4(p,1,:,:)=squeeze(nanmean(lf_data{sel_pat}.face_miss));  
        else
        hf_pow1(p,1,:,:)=hf_data{sel_pat}.word_hit; 
        hf_pow2(p,1,:,:)=hf_data{sel_pat}.word_miss; 
        hf_pow3(p,1,:,:)=hf_data{sel_pat}.face_hit; 
        hf_pow4(p,1,:,:)=hf_data{sel_pat}.face_miss;
        lf_pow1(p,1,:,:)=lf_data{sel_pat}.word_hit; 
        lf_pow2(p,1,:,:)=lf_data{sel_pat}.word_miss; 
        lf_pow3(p,1,:,:)=lf_data{sel_pat}.face_hit; 
        lf_pow4(p,1,:,:)=lf_data{sel_pat}.face_miss;  
        end
    end
    
   f1=nearest(lf_data1.freq,foi(1));
   f2=nearest(lf_data1.freq,foi(2));
   t1=nearest(lf_data1.time,toi1(1));
   t2=nearest(lf_data1.time,toi1(2));
      
   t3=nearest(lf_data1.time,toi2(1));
   t4=nearest(lf_data1.time,toi2(2));
  cond1=squeeze(nanmean(nanmean(lf_pow1(:,:,f1:f2,t1:t2),3),4));
  cond2=squeeze(nanmean(nanmean(lf_pow2(:,:,f1:f2,t1:t2),3),4));
  cond3=squeeze(nanmean(nanmean(lf_pow3(:,:,f1:f2,t1:t2),3),4));
  cond4=squeeze(nanmean(nanmean(lf_pow4(:,:,f1:f2,t1:t2),3),4));  
  
  cond5=squeeze(nanmean(nanmean(lf_pow1(:,:,f1:f2,t3:t4),3),4));
  cond6=squeeze(nanmean(nanmean(lf_pow2(:,:,f1:f2,t3:t4),3),4));
  cond7=squeeze(nanmean(nanmean(lf_pow3(:,:,f1:f2,t3:t4),3),4));
  cond8=squeeze(nanmean(nanmean(lf_pow4(:,:,f1:f2,t3:t4),3),4)); 
  
  % avg word and face
  avg_pow=[(cond1+cond3).*0.5,(cond2+cond4).*0.5,(cond5+cond7).*0.5,(cond6+cond8).*0.5];
    conditions={'rem','forg','rem','forg'}

fig=figure
  subplot(2,2,1)
  hold on
  bar([nanmean(avg_pow)])
  ylim([-0.1 0.2])  
  scatter(reshape(repmat(1:4,size(avg_pow,1),1),[],1),reshape(avg_pow,[],1),'k');
  ax=gca;
  ax.XTick=(1:4);
  ax.XTickLabel=conditions;
  title(strcat('gamma in ',totest,' theta electrodes'))

        
    subplot(2,2,2)
    % plot tf 
    [~,~,~,tstat]=ttest((hf_pow1+hf_pow3).*0.5,(hf_pow2+hf_pow4).*0.5); 

    toi_plot=[0,1.5];
    tp1=nearest(hf_data1.time,toi_plot(1));
    tp2=nearest(hf_data1.time,toi_plot(2));
    foi_plot=[30 100];
    fp1=nearest(hf_data1.freq,foi_plot(1));
    fp2=nearest(hf_data1.freq,foi_plot(2));

    imagesc(hf_data1.time(tp1:tp2),hf_data1.freq(fp1:fp2),squeeze(tstat.tstat(:,:,fp1:fp2,tp1:tp2)),[-4 4]);
    set(gca, 'Ydir','normal')
    colormap('jet')
    colorbar
    
     subplot(2,2,4)
    % plot tf 
    [~,~,~,tstat]=ttest((lf_pow1+lf_pow3).*0.5,(lf_pow2+lf_pow4).*0.5); 

    toi_plot=[0,1.5];
    tp1=nearest(lf_data1.time,toi_plot(1));
    tp2=nearest(lf_data1.time,toi_plot(2));
    foi_plot=[1 30];
    fp1=nearest(lf_data1.freq,foi_plot(1));
    fp2=nearest(lf_data1.freq,foi_plot(2));

    imagesc(lf_data1.time(tp1:tp2),lf_data1.freq(fp1:fp2),squeeze(tstat.tstat(:,:,fp1:fp2,tp1:tp2)),[-4 4]);
    set(gca, 'Ydir','normal')
    colormap('jet')
    colorbar
    
   savefig(fig,fullfile(path_figs,fig_number)); 
  


