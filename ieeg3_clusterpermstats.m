%% permute number of electrodes: ANOVA

project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20180528\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%%%%%%%%%%%%%%%%%  fix electrode positions: use elecinfo correct!


%% construct neighbourhood structure
% 
path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(path_in,'stats');
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};

freq_bands={'theta'}; % check in one frequency band for trial numbers


for f=1:numel(freq_bands)
    % define depending on freq band
    % tf for finding electrodes
        sel_freq_band=freq_bands{f};
        effects={'sme_neg','sme_pos','smeword<smeface'};
        switch sel_freq_band
            case 'alphabeta'
                foi        = [8 20];
                toi     = [0.3 1.5];
                fig_numbers='suppfig3_alphabeta';
                freq_def='lf';
            case 'theta'
                foi        = [2 5];
                toi     = [1 1.5];
                fig_numbers='suppfig3_theta';
                freq_def='lf';
            case 'gamma'
                foi        = [50 90];
                toi     = [0.3 1];
                fig_numbers='fig4g_gamma';
                freq_def='hf';
        end 
% 
for n=1:numel(pat)
% 
%load data
load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_faces')))
data1=freq;
load(fullfile(path_in,strcat(pat{n},'_',freq_def,'_words')))
data2=freq;
clear freq
%    
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

 freq1=squeeze(nanmean(nanmean(data1.powspctrm(hitind1,:,f1:f2,t1:t2),3),4)); 
 freq2=squeeze(nanmean(nanmean(data1.powspctrm(missind1,:,f1:f2,t1:t2),3),4));   
 freq3=squeeze(nanmean(nanmean(data2.powspctrm(hitind2,:,f1:f2,t1:t2),3),4)); 
 freq4=squeeze(nanmean(nanmean(data2.powspctrm(missind2,:,f1:f2,t1:t2),3),4));  
%  
hit1=sum(~isnan(freq1));
hit2=sum(~isnan(freq2));
miss1=sum(~isnan(freq3));
miss2=sum(~isnan(freq4));
%  
elec_trial_num_ok=(hit1>=10 & hit2>=10 & miss1>=10 &miss2>=10);

clear f1 f2 t1 t2 hit1 hit2 miss1 miss2 freq1 freq2 freq3 freq4  hitind1 hitind2 missind1 missind2 data2

 ok_elec=find(elec_trial_num_ok==1);

e_info{n}.label=data1.label(ok_elec);
e_info{n}.elec_pos=data1.elecinfo.elecpos_bipolar(ok_elec,:);
clear  ok_elec elec_trial_num_ok 
end


% define neighbours: get electrodes with all info

radius_clus=20; % radius for pure distance clustering
% 
 mri=ft_read_mri(fullfile(project_path,'scripts','additional_scripts','T1.nii'));
 
 for n=1:numel(pat)
     for e=1:numel(e_info{n}.label)
cfg=[];
cfg.roi=e_info{n}.elec_pos(e,:);
cfg.inputcoord          = 'mni' ;
cfg.sphere= radius_clus;
mask_clus= ft_volumelookup(cfg, mri);
cluster_ind=find(mask_clus);

  e_info{n}.cluster_ind{e}=cluster_ind;
     end
 end
 
 % put all info in one structure  (some extra resorting, could be
 % programmed better...)
 
 
 electrode_info=[];

 electrode_info.pat=[];
 electrode_info.mni=[];
 electrode_info.name=[];

 
electrode_info.cluster_ind=[];
for n=1:numel(pat)
num_elec=numel(e_info{n}.label);
pat_tmp=repmat(pat(n),num_elec,1);
mni_tmp=e_info{n}.elec_pos;
name_tmp=e_info{n}.label;
cluster_tmp=e_info{n}.cluster_ind;


 electrode_info.pat=[electrode_info.pat;pat_tmp];
 electrode_info.mni=[electrode_info.mni;mni_tmp];
 electrode_info.name=[electrode_info.name;name_tmp];
electrode_info.cluster_ind=[electrode_info.cluster_ind;cluster_tmp'];
end
 
% matrix

 % for every electrode check which are neighbouring electrodes
  neighbourhood_distance=zeros(numel(electrode_info.name));
 for e1=1:numel(electrode_info.name)
 for e2=1:numel(electrode_info.name)
   neighbourhood_distance(e1,e2)=~isempty(intersect(electrode_info.cluster_ind{e1},electrode_info.cluster_ind{e2}));
 end
 end


electrode_info.neighbourhood_noanatomyradius20mm=neighbourhood_distance;

save(fullfile(path_out,'cluster_electrode_info.mat'),'electrode_info')
end
 
%% permute number of electrodes: ANOVA
path_figs=fullfile(project_path,'figures');
% 
path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(path_in,'stats');
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};

freq_bands={'theta','alphabeta','gamma'}; % check in one frequency band for trial numbers


for sel_f=1:numel(freq_bands)
    % define depending on freq band
    % tf for finding electrodes
        sel_freq_band=freq_bands{sel_f};
        effects={'sme_neg','sme_pos','smeword<smeface'};
        switch sel_freq_band
            case 'alphabeta'
                foi        = [8 20];
                toi     = [0.3 1.5];
                fig_numbers='suppfig3_alphabeta';
                freq_def='lf';
            case 'theta'
                foi        = [2 5];
                toi     = [1 1.5];
                fig_numbers='suppfig3_theta';
                freq_def='lf';
            case 'gamma'
                foi        = [50 90];
                toi     = [0.3 1];
                fig_numbers='suppfig3_gamma';
                freq_def='hf';
        end 
        
        

alpha_level=0.05;

% neighbourhood structure!
load(fullfile(path_out,'cluster_electrode_info.mat'))

neighbourhood_type='distance20';
switch neighbourhood_type 
    case  'distance20'
  neighbourhood=electrode_info.neighbourhood_noanatomyradius20mm;
      
end
effects={'mem', 'cond', 'interaction'};
min_t=10;
nrand=1000;



%%%%% stats data
p_all=[];
f_all=[];
h_all=[];
dir_all=[];
diff_all=[];
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

 
 freq1=squeeze(nanmean(nanmean(data1.powspctrm(hitind1,:,f1:f2,t1:t2),3),4)); 
 freq2=squeeze(nanmean(nanmean(data1.powspctrm(missind1,:,f1:f2,t1:t2),3),4));   
 freq3=squeeze(nanmean(nanmean(data2.powspctrm(hitind2,:,f1:f2,t1:t2),3),4)); 
 freq4=squeeze(nanmean(nanmean(data2.powspctrm(missind2,:,f1:f2,t1:t2),3),4));  
 
 hit1=sum(~isnan(freq1));
hit2=sum(~isnan(freq2));
miss1=sum(~isnan(freq3));
miss2=sum(~isnan(freq4));
 
 elec_trial_num_ok=(hit1>=10 & hit2>=10 & miss1>=10 &miss2>=10);


ok_elec=find( elec_trial_num_ok==1);

 freq1=freq1(:,ok_elec);
 freq2=freq2(:,ok_elec);
 freq3=freq3(:,ok_elec);
 freq4=freq4(:,ok_elec);
 

 freq=[freq1;freq2;freq3;freq4];

mem=[repmat({'hit'},size(hitind1));repmat({'miss'},size(missind1));repmat({'hit'},size(hitind2));repmat({'miss'},size(missind2))];
cond=[repmat({'word'},size([hitind1;missind1]));repmat({'face'},size([hitind2;missind2]))];
  
    for e=1:size(freq1,2)

        [p_tmp,tbl,stats] = anovan(freq(:,e),{mem cond},'model','interaction','varnames',{'mem','cond'},'display','off');               
        h(e,:)=double(p_tmp'<alpha_level);
        p(e,:)=double(p_tmp');
        f(e,:)=[tbl{2:4,6}];
       % 1:mem, 2:cond, 3:interaction
           
    end
     diff(:,1)=((((nanmean(freq1)+nanmean(freq3)).*0.5)-((nanmean(freq2)+nanmean(freq4)).*0.5)));
     diff(:,2)=((((nanmean(freq1)+nanmean(freq2)).*0.5)-((nanmean(freq3)+nanmean(freq4)).*0.5)));
     diff(:,3)=(((nanmean(freq1)-nanmean(freq2))-(nanmean(freq3)-nanmean(freq4))));

    
     direction(:,1)=sign((((nanmean(freq1)+nanmean(freq3)).*0.5)-((nanmean(freq2)+nanmean(freq4)).*0.5)));
     direction(:,2)=sign((((nanmean(freq1)+nanmean(freq2)).*0.5)-((nanmean(freq3)+nanmean(freq4)).*0.5)));
     direction(:,3)=sign(((nanmean(freq1)-nanmean(freq2))-(nanmean(freq3)-nanmean(freq4))));

p_all=[p_all;p];
f_all=[f_all;f];
h_all=[h_all;h];
dir_all=[dir_all; direction];
diff_all=[diff_all;diff];

freq1_all{n}=freq1;
freq2_all{n}=freq2;
freq3_all{n}=freq3;
freq4_all{n}=freq4;

mem_all{n}=mem;
cond_all{n}=cond;

clear direction h p_tmp p f tbl stats hitind1 hitind2 missind1 missind2 diff
clear freq1 freq2 freq3 freq4

end

clear hit1 hit2 miss1 miss2 elec_trial_num_ok data1 pos_e sel_e data2 cond mem freq f1 f2 t1 t2

dir_h=h_all.*dir_all;

directions={'pos','neg'};
% build subfunction for clustering: input sig_vec, neighboorhood mat,
% output mat: num cluster x cluster 0/1 each electrodes

for eff=1:numel(effects)
    for d=1:numel(directions)
      direction_eff=  directions{d};
      switch direction_eff
          case 'pos'
         vec_h=(dir_h(:,eff)==1); % vector  significant electrodes
        [num_cluster, clustermat]=diy_neighbourcluster(vec_h,neighbourhood);
    % caclulate cluster sums
    posclustersum{eff}=sum(repmat(f_all(:,eff),1,num_cluster).*clustermat);
    posclusterlabelmat{eff}=clustermat;
          case 'neg'
         vec_h=(dir_h(:,eff)==-1); % vector  significant electrodes
        [num_cluster, clustermat]=diy_neighbourcluster(vec_h,neighbourhood); 
    % caclulate cluster sums
    negclustersum{eff}=sum(repmat(f_all(:,eff),1,num_cluster).*clustermat);
    negclusterlabelmat{eff}=clustermat;
    end
    end
end

p_data=p_all;
f_data=f_all;
h_data=h_all;
dir_data=dir_all;

neg_size=max(cellfun(@numel,negclustersum));
pos_size=max(cellfun(@numel,posclustersum));
all_elec=size(h_data,1);
   randposclustersum=zeros(numel(effects),nrand,pos_size);
   randposclusterlabelmat=zeros(numel(effects),nrand,all_elec,pos_size);
   randnegclustersum=zeros(numel(effects),nrand,neg_size);
   randnegclusterlabelmat=zeros(numel(effects),nrand,all_elec,neg_size);
   
%%%%%% ranodmisation
 for r=1:nrand
   p_all=[];
f_all=[];
h_all=[];
dir_all=[];
tic
  for n=1:numel(pat) 
      freq1=freq1_all{n};
      freq2=freq2_all{n};
      freq3=freq3_all{n};
      freq4=freq4_all{n};
     
      freq=[freq1;freq2;freq3;freq4]; 
      mem=mem_all{n};
      cond=cond_all{n};
      
      rand_ind= randperm(numel(mem));
      mem_rand=mem(rand_ind);
      rand_ind= randperm(numel(mem));
      cond_rand=cond(rand_ind);
             
      hitind1=find(strcmp('word',cond_rand)&strcmp('hit',mem_rand));
      hitind2=find(strcmp('face',cond_rand)&strcmp('hit',mem_rand));
      missind1=find(strcmp('word',cond_rand)&strcmp('miss',mem_rand));
      missind2=find(strcmp('face',cond_rand)&strcmp('miss',mem_rand));
      
      direction(:,1)=sign((((nanmean(freq(hitind1,:))+nanmean(freq(missind1,:))).*0.5)-((nanmean(freq(hitind2,:))+nanmean(freq(missind2,:))).*0.5)));
      direction(:,2)=sign((((nanmean(freq(hitind1,:))+nanmean(freq(hitind2,:))).*0.5)-((nanmean(freq(missind1,:))+nanmean(freq(missind2,:))).*0.5)));
      direction(:,3)=sign(((nanmean(freq(hitind1,:))-nanmean(freq(missind1,:)))-(nanmean(freq(hitind2,:))-nanmean(freq(missind2,:)))));

        for e=1:size(freq1,2)
           [p_tmp,tbl,stats] = anovan(freq(:,e),{mem_rand cond_rand},'model','interaction','varnames',{'mem','cond'},'display','off');               
           h(e,:)=double(p_tmp'<alpha_level); 
           f(e,:)=[tbl{2:4,6}];
           p(e,:)=double(p_tmp');
            % 1:mem, 2:cond, 3:interaction
        end
    p_all=[p_all;p];
    f_all=[f_all;f];
    h_all=[h_all;h];
    dir_all=[dir_all; direction];
    clear direction h p_tmp p f tbl stats hitind1 hitind2 missind1 missind2
    clear freq1 freq2 freq3 freq4
  end
    
 dir_h=h_all.*dir_all;

directions={'pos','neg'};

   
  for eff=1:numel(effects)
    for d=1:numel(directions)
      direction_eff=  directions{d};
      switch direction_eff
          case 'pos'
         vec_h=(dir_h(:,eff)==1); % vector  significant electrodes
        [num_cluster, clustermat]=diy_neighbourcluster(vec_h,neighbourhood);
    % caclulate cluster sums

    if num_cluster<pos_size
        clustermat=[clustermat,zeros(all_elec,pos_size-num_cluster)];
    end
    randposclustersum(eff,r,:)=sum(repmat(f_all(:,eff),1,pos_size).*clustermat(:,1:pos_size));
    randposclusterlabelmat(eff,r,:,:)=clustermat(:,1:pos_size);
          case 'neg'
         vec_h=(dir_h(:,eff)==-1); % vector  significant electrodes
     
         [num_cluster, clustermat]=diy_neighbourcluster(vec_h,neighbourhood); 
    % caclulate cluster sums
    if num_cluster<neg_size
        clustermat=[clustermat,zeros(all_elec,neg_size-num_cluster)];
    end
    
    randnegclustersum(eff,r,:)=sum(repmat(f_all(:,eff),1,neg_size).*clustermat(:,1:neg_size));
    randnegclusterlabelmat(eff,r,:,:)=clustermat(:,1:neg_size);
      end
    end
  end
  r
  toc
 end

 % check for each effect the randdistributions
 
  for eff=1:numel(effects)
  sort_dist=sort(squeeze(randnegclustersum(eff,:,:)));
  
  data_dist=negclustersum{eff};
  check=1;
   ind=0;
   clear p_neg
  while check==1
     ind=ind+1;
       p_neg(ind)= (nrand-nearest(sort_dist(:,ind),data_dist(ind)))./nrand;
  check=(p_neg(ind)<(0.05/(ind)) & ind<numel(data_dist));
  end
    p_neg_all{eff}=p_neg;
  
  end
 
 for eff=1:numel(effects)
  sort_dist=sort(squeeze(randposclustersum(eff,:,:)));
  
  data_dist=posclustersum{eff};
  check=1;
   ind=0;
  clear p_pos
   while check==1
     ind=ind+1;
       p_pos(ind)= (nrand-nearest(sort_dist(:,ind),data_dist(ind)))./nrand;
  check=(p_pos(ind)<(0.05/(ind)) & ind<numel(data_dist));
  end
    p_pos_all{eff}=p_pos;
  
 end
 
 
 permstats.effects=effects;
 permstats.f=f_data;
 permstats.h=h_data;
 permstats.p=p_data;
 permstats.dir=dir_data;
 permstats.nrand=nrand;
 permstats.frq=foi;
 permstats.time=toi;
 permstats.electrodes=electrode_info;
 permstats.negclusterlabelmat=negclusterlabelmat;
 permstats.negclustersum=negclustersum;
 permstats.posclusterlabelmat=posclusterlabelmat;
 permstats.posclustersum=posclustersum;
 permstats.posclusterprob=p_pos_all;
 permstats.negclusterprob=p_neg_all;
 permstats.diffdata=diff_all;

 
clear clustermat cond_rand data_dist diff_all dir_all dir_data dir_h f_all f_data freq 
clear negclusterlabelmat negclustersum p_all p_data p_neg p_neg_all p_pos p_pos_all
clear freq1_all freq2_all freq3_all freq4_all h_all h_data neg_size vec_h num_cluster sort_dist pos_size
clear posclusterlabelmat posclustersum  randposclusterlabelmat randposclustersum randnegclusterlabelmat randnegclustersum

 % save stats
  freq_str=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
 time_str=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms');
 
 
% % plot electrodes (note these figures are slightly different from in the
% manuscript (manuscript figure are plotted with caret)) here using new
% fieldtrip ecog toolboxextension
alpha_toplot=0.1; %plot all clusters with p smaller 0.1
for sel_eff=1: numel(effects)
allstat.all_elec=permstats.electrodes.name;
allstat.all_pos=permstats.electrodes.mni;
allstat.all_pat=permstats.electrodes.pat;
    % combine cluster info to sig neg/pos cluster vector
     % neg vector
        check_negcluster_sig=permstats.negclusterprob{sel_eff}<alpha_toplot;
        check_negcluster_sum=permstats.negclustersum{sel_eff}(1:numel(check_negcluster_sig))>0;
        check_negcluster=find(check_negcluster_sig&check_negcluster_sum);
        hs=zeros(numel(permstats.electrodes.name),1);
        if ~isempty(check_negcluster)
        hs=hs-(sum(permstats.negclusterlabelmat{sel_eff}(:,check_negcluster),2));
        end       
        % pos vector
        check_poscluster_sig=permstats.posclusterprob{sel_eff}<alpha_toplot;
        check_poscluster_sum=permstats.posclustersum{sel_eff}(1:numel(check_poscluster_sig))>0;
        check_poscluster=find(check_poscluster_sig&check_poscluster_sum);       
        if ~isempty(check_poscluster)
        hs=hs+(sum(permstats.posclusterlabelmat{sel_eff}(:,check_poscluster),2));
        end

allstat.all_dir=hs;
if sel_eff==3
allstat.all_dir=allstat.all_dir.*-1; % contrast is faces vs words, interaction is plotted words vs faces
end
allstatplot_ft2019(allstat,'cluster',path_figs,strcat(fig_numbers,effects{sel_eff}),project_path,fieldtrip_path);
end
% for reproducing caret plots use the here exported foci/ foci color files
% and plot foci on spm surface in caret
path_foci=fullfile(path_out,'foci');
mkdir(path_foci)
file_name= strcat(sel_freq_band,freq_str,time_str);
allstat2caretfoci(permstats,'cluster',pat,path_foci,file_name);

save(fullfile(path_out,strcat('cluster_rand',neighbourhood_type, num2str(alpha_level*100),'%',freq_str,time_str)),'permstats')
end
 
