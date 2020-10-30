%

% figure 2 g&h, supplemental figure 2
% chi square different thresholds
% correlation

project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));
    
       
%% get iEEG electrodes in sourceclusters
path_in=fullfile(project_path,'ieeg_data','freq');
path_stats_MEG=fullfile(project_path,'meg_data','source','stats');
path_stats_iEEG=fullfile(project_path,'ieeg_data','freq','stats');

path_out=fullfile(project_path,'overlap_ieeg_meg');

mkdir(path_out)

freq_oi={'alphabeta','gamma'};

stat_file_meg={'sourcestats_freqmain_cond8to20Hztime300to1500ms';
           'sourcestats_freqmain_cond50to90Hztime300to1000ms'}
       
stat_file_ieeg={'allelecstatcondalphabeta8to20Hz300to1500ms';
           'allelecstatcondgamma50to90Hz300to1000ms'}
       
       
all_meg_def={'sig_cluster','direction'};

% all patients with word and face session
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};%

  % load template mni brain (to interpolate for mni coordinate check)
mri_file=fullfile(project_path,'scripts','additional_scripts','single_subj_T1_1mm.nii');
mri=ft_read_mri(mri_file);


% first step get all electrode coordinates (pat number, elec name, mni)

for def=1:numel(all_meg_def)
meg_def=all_meg_def{def}
for f=1:numel(freq_oi)
    sel_freq=freq_oi{f};

    load(fullfile(path_stats_iEEG,stat_file_ieeg{f}))
    
    e_info=cell(0,3);
    directed_h_all=[];
    direction_all=[];
    mean_diff_all=[];

for n=1:numel(allstat)
    pat_def=repmat(pat(n),size(allstat{n}.elecs));
   e_info=[e_info;pat_def,allstat{n}.elecs,num2cell(allstat{n}.elecpos,2)];
   mean_diff=((allstat{n}.all_freq_avg(:,1)+allstat{n}.all_freq_avg(:,2)).*0.5)-((allstat{n}.all_freq_avg(:,3)+allstat{n}.all_freq_avg(:,4)).*0.5);
   mean_diff_all=[mean_diff_all;mean_diff];
   directed_h=(allstat{n}.h.*sign(allstat{n}.f))';
   direction_e=sign(mean_diff);
   directed_h_all=[directed_h_all;directed_h];
   direction_all=[direction_all;direction_e];
end
ok_elec=~isnan(directed_h_all);
e_info=e_info(ok_elec,:);
directed_h_all=directed_h_all(ok_elec,:);
direction_all=direction_all(ok_elec,:);
mean_diff_all=mean_diff_all(ok_elec,:);
   %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
   
    ieeg_in_meg_roi.all_ok_elec=e_info;
    ieeg_in_meg_roi.stats=stat_file_meg{f};
    ieeg_in_meg_roi.effects='cond';
    ieeg_in_meg_roi.directed_h_all=directed_h_all;
    ieeg_in_meg_roi.direction_all=direction_all;
    ieeg_in_meg_roi. mean_diff_all= mean_diff_all;
    
  % load sourcestats of interest
  load(fullfile(path_stats_MEG,stat_file_meg{f}));


  cfg=[];              
  cfg.parameter = 'avg';                
  avgint=ft_sourceinterpolate(cfg, stat,mri);

 all_pos=round(reshape([e_info{:,3}],3,[])');

 [x,epos_e_info,epos_instat]=intersect(all_pos,avgint.pos,'rows','stable');
 % to index back to e_info
 avg_inelec=avgint.avg(epos_instat);
  ieeg_in_meg_roi.avg_effect=avg_inelec;
  % interpolation
  switch meg_def
      case 'sig_cluster'
        if isfield(stat,'negclusters')
              if ~isempty(stat.negclusters)
                  if stat.negclusters(1).prob<=0.05
       %               hm_t_neg=min(stat.stat).*cutoff;
                      sig_clusters=find([stat.negclusters.prob]<=0.05);
                   % sig_clusters=find([stat.negclusters.prob]);
                      mask_neg=(stat.negclusterslabelmat>0 & stat.negclusterslabelmat<=sig_clusters(end) );
                      negstat=stat;
                      negstat.negclusterslabelmat=mask_neg;


              cfg=[];
                cfg.parameter = 'negclusterslabelmat';
                negstatint=ft_sourceinterpolate(cfg, negstat,mri);

                
                % all elecpos mat
                [x,epos_e_info,epos_instat]=intersect(all_pos,negstatint.pos,'rows','stable');
                % to index back to e_info
                in_cluster_check=negstatint.negclusterslabelmat(epos_instat)>0;
                %plot electrodes on brain to check
                negstatintplot=negstatint;
                negstatintplot.negclusterslabelmat(epos_instat)=10; % setting to plot
     
            ieeg_in_meg_roi.elec_in_negcluster=in_cluster_check;
        else
            ieeg_in_meg_roi.elec_in_negcluster=[];
               end
            end
         end


        if isfield(stat,'posclusters')
              if ~isempty(stat.posclusters)
                  if stat.posclusters(1).prob<=0.05
                     % hm_t_pos=max(stat.stat).*cutoff;
                      sig_clusters=find([stat.posclusters.prob]<=0.05);
                      %sig_clusters=find([stat.posclusters.prob]);

                      mask_pos=(stat.posclusterslabelmat>0 & stat.posclusterslabelmat<=sig_clusters(end));
                      posstat=stat;
                      posstat.posclusterslabelmat=mask_pos;


                 cfg=[];
                cfg.parameter = 'posclusterslabelmat';
                posstatint=ft_sourceinterpolate(cfg, posstat,mri);

                % all elecpos mat
                all_pos=round(reshape([e_info{:,3}],3,[])');

                [x,epos_e_info,epos_instat]=intersect(all_pos,posstatint.pos,'rows','stable');
                % to index back to e_info

                in_cluster_check=posstatint.posclusterslabelmat(epos_instat)>0;
                %plot electrodes on brain to check
                posstatintplot=posstatint;
                posstatintplot.posclusterslabelmat(epos_instat)=10;  % setting to plot
                % plot to check
%                 cfg = [];
%                 %cfg.method        = 'slice';
%                 cfg.method        = 'ortho';
%                 cfg.funparameter  = 'posclusterslabelmat';
%                 cfg.maskparameter = cfg.funparameter;
%                 %cfg.funcolorlim   = [-1000000 0];
%                 %cfg.opacitylim    = [-1000000 -10000]; 
%                 %cfg.opacitymap    = 'rampup';  
%                 figure
%                 ft_sourceplot(cfg, posstatintplot);
            ieeg_in_meg_roi.elec_in_poscluster=in_cluster_check;
        else
            ieeg_in_meg_roi.elec_in_poscluster=[];
                  end
              end
              end
      case 'direction'

                mask_neg=(stat.stat<0);
                negstat=stat;
                negstat.negclusterslabelmat=mask_neg;


              cfg=[];
                cfg.parameter = 'negclusterslabelmat';
                negstatint=ft_sourceinterpolate(cfg, negstat,mri);
                % all elecpos mat
                all_pos=round(reshape([e_info{:,3}],3,[])');
                [x,epos_e_info,epos_instat]=intersect(all_pos,negstatint.pos,'rows','stable');
                % to index back to e_info
                in_cluster_check=negstatint.negclusterslabelmat(epos_instat)>0;
                %plot electrodes on brain to check
                negstatintplot=negstatint; 
                negstatintplot.negclusterslabelmat(epos_instat)=10;  % setting to plot
     
            ieeg_in_meg_roi.elec_in_negcluster=in_cluster_check;
     


                      mask_pos=(stat.stat>0);
                      posstat=stat;
                      posstat.posclusterslabelmat=mask_pos;


                 cfg=[];
                cfg.parameter = 'posclusterslabelmat';
                posstatint=ft_sourceinterpolate(cfg, posstat,mri);

                % all elecpos mat
                all_pos=round(reshape([e_info{:,3}],3,[])');

                [x,epos_e_info,epos_instat]=intersect(all_pos,posstatint.pos,'rows','stable');
                % to index back to e_info

                in_cluster_check=posstatint.posclusterslabelmat(epos_instat)>0;
                %plot electrodes on brain to check
            %     posstatintplot=posstatint;
            %     posstatintplot.posclusterslabelmat(epos_instat)=10;
            %     % 
            %     cfg = [];
            %     %cfg.method        = 'slice';
            %     cfg.method        = 'ortho';
            %     cfg.funparameter  = 'posclusterslabelmat';
            %     cfg.maskparameter = cfg.funparameter;
            %     %cfg.funcolorlim   = [-1000000 0];
            %     %cfg.opacitylim    = [-1000000 -10000]; 
            %     %cfg.opacitymap    = 'rampup';  
            %     figure
            %     ft_sourceplot(cfg, posstatintplot);
            ieeg_in_meg_roi.elec_in_poscluster=in_cluster_check;
   
  end  


  save(fullfile(path_out,strcat('elecs_in_megroi_',sel_freq,'_cutoff_',meg_def,'.mat')), 'ieeg_in_meg_roi')
end
end  

  
  
 
  %% do chi square
 %  next step: get anova values for each electrode
path_stats=fullfile(project_path,'overlap_ieeg_meg');
path_figs=fullfile(project_path,'figures');

  

all_freqs={'alphabeta','gamma'}

         
all_meg_def={'sig_cluster','direction'};
for def=1:numel(all_meg_def)
meg_def=all_meg_def{def}
switch meg_def
    case 'sig_cluster'
        fig_defs={'fig2g','fig2h'};
    case 'direction'
        fig_defs={'supfig2a','supfig2c'};
end
    
 for f=1:numel(all_freqs)
  freq=all_freqs{f};
 load(fullfile(path_stats,strcat('elecs_in_megroi_',freq,'_cutoff_',meg_def,'.mat')))
 
 
  contrasts={ieeg_in_meg_roi.stats};
  effects={ieeg_in_meg_roi.effects};
  poscluster=ieeg_in_meg_roi.elec_in_poscluster;
  negcluster=ieeg_in_meg_roi.elec_in_negcluster;
  
pos=~isempty(ieeg_in_meg_roi.elec_in_poscluster)
neg=~isempty(ieeg_in_meg_roi.elec_in_negcluster)
 
if numel(pos)<numel(contrasts)
tmp=zeros(1, numel(contrasts)-numel(pos));
pos=[pos,tmp];
end

if numel(neg)<numel(contrasts)
tmp=zeros(1, numel(contrasts)-numel(neg));
neg=[neg,tmp];
end

clusters=[pos;neg];


for c=1:numel(contrasts)
          for e=1:numel(effects)
          if sum(clusters(:,c))==2
           [tbl,chi2,p,labels]  = crosstab(ieeg_in_meg_roi.directed_h_all(:,e),ieeg_in_meg_roi.elec_in_poscluster-ieeg_in_meg_roi.elec_in_negcluster); 
   
         display(strcat(freq,'iEEG:', effects{e},'x MEG:p-value', num2str(p),'chi^2=', num2str(chi2)))
          elseif sum(clusters(1,c))==1 & sum(clusters(2,c))==0
            [tbl,chi2,p,labels]  = crosstab(ieeg_in_meg_roi.directed_h_all(:,e),ieeg_in_meg_roi.elec_in_poscluster) ;
             display(strcat(freq,'iEEG:', effects{e},'x MEG:p-value', num2str(p),'chi^2=', num2str(chi2)))

           elseif sum(clusters(1,c))==0 & sum(clusters(2,c))==1
            [tbl,chi2,p,labels]  = crosstab((ieeg_in_meg_roi.directed_h_all(:,e)),ieeg_in_meg_roi.elec_in_negcluster.*-1) ;
           display(strcat(freq,'iEEG:', effects{e},'x MEG:p-value', num2str(p),'chi^2=', num2str(chi2)))
              
          end
          % table organisation: rows: iEEG effect direction, columns: MEG area definition
          
           if sum(clusters(:,c))~=0
          if p<0.05
              fig=figure
 
              
              subplot(1,2,1)
              axis off
              text(0.5,0.9,strcat('MEG:'))
              text(0,0.8,'iEEG')
              ind1=find(cellfun(@isempty,labels(:,2))==0);
              table_line1=[NaN,cell2double(labels(ind1,2))'];
              ind2=find(cellfun(@isempty,labels(:,1))==0);
              
              table_otherlines=[cell2double(labels(ind2,1)),tbl];
              text(0.3,0.8,num2str([table_line1;table_otherlines]))
               
               title (strcat(freq,'iEEG:', effects{e},'x MEG:p-value', num2str(p),'chi^2=', num2str(chi2)))

               
               % calculate expected
%                sum_col=sum(tbl);
%                sum_row=sum(tbl,2);
%                sum_all=sum(sum_col);
%                rel_row=sum_row./sum_all;
%                rel_col=sum_col./sum_all; 
%                expected=(rel_row*rel_col).*sum_all;
%                 
%                text(0,0.5,num2str(expected))


         
                          subplot(1,2,2)

                for row=1:size(tbl,1)
                  tbl3(row,:)=tbl(row,:)./sum(tbl(row,:));
                end
                % resort to alway plot zero on top: row 3 has to be row 2
                 bar(tbl3,'stacked','DisplayName','table') 
                ax=gca;
                ax.XAxis.TickValues = 1:size(labels,1);
                ax.XAxis.TickLabels = labels(:,2);    
                clear tbl3
          
          end
           end
          end
          savefig(fig,fullfile(path_figs,fig_defs{f}))
close all 

end
 end
end

   %% correlate avg pow meg with avg power ieeg
path_stats=fullfile(project_path,'overlap_ieeg_meg');
path_figs=fullfile(project_path,'figures');  
all_freqs={'alphabeta','gamma'}      
all_meg_def={'sig_cluster'};
for def=1:numel(all_meg_def)
meg_def=all_meg_def{def}
 fig_defs={'supfig2b','supfig2d'};

    
 for f=1:numel(all_freqs)
  freq=all_freqs{f};
 load(fullfile(path_stats,strcat('elecs_in_megroi_',freq,'_cutoff_',meg_def,'.mat')))
 

 fig=figure         
[RHO,PVAL] = corr(ieeg_in_meg_roi.avg_effect.*-1,ieeg_in_meg_roi.mean_diff_all.*-1,'type','Spearman','rows','complete')

scatterhist(ieeg_in_meg_roi.avg_effect.*-1,ieeg_in_meg_roi.mean_diff_all.*-1,'Direction','out','NBins',[20,20])
ylabel('word-face iEEG: z-value diff')
xlabel('word-face MEG: z-value diff')
title([{'correlation iEEG electrodes with power difference in MEG'};{strcat('spearman rho=', num2str(RHO),' P=',num2str(PVAL))}])
savefig(fig,fullfile(path_figs,strcat(fig_defs{f},'correlation')))
close all
 end
end