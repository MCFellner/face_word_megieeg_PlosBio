project_path='D:\face_word\faceword_shareddata\';
fieldtrip_path='D:\matlab_tools\fieldtrip-20160122\';

addpath (fieldtrip_path)
ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts'));


%% permutation statistic: number of electrodes
path_in=fullfile(project_path,'ieeg_data','freq');
path_out=fullfile(path_in,'stats');
path_figs=fullfile(project_path,'figures');


mkdir (path_out);
% all patients with word and face session
pat={'pat02','pat04','pat05','pat08','pat10','pat11','pat15','pat16','pat17','pat19','pat20','pat21','pat22'};
freq_bands={'theta','alphabeta','gamma'};

alpha_level=0.05;

min_t=10;
nrand=1000;

for f=1:numel(freq_bands)
    % define depending on freq band
    % tf for finding electrodes
        sel_freq_band=freq_bands{f};
        effects={'sme_neg','sme_pos','smeword<smeface'};
        switch sel_freq_band
            case 'alphabeta'
                foi        = [8 20];
                toi     = [0.3 1.5];
                fig_numbers='fig4g_alphabeta';
                freq_def='lf';
            case 'theta'
                foi        = [2 5];
                toi     = [1 1.5];
                fig_numbers='fig4g_theta';
                freq_def='lf';
            case 'gamma'
                foi        = [50 90];
                toi     = [0.3 1];
                fig_numbers='fig4g_gamma';
                freq_def='hf';
        end  
        
        

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
 
 % remove electrodes with not enough trials
    hit1=sum(~isnan(freq1));
    hit2=sum(~isnan(freq2));
    miss1=sum(~isnan(freq3));
    miss2=sum(~isnan(freq4));
 
 ok_elec=(hit1>=10 & hit2>=10 & miss1>=10 &miss2>=10);
 freq1=freq1(:,ok_elec);
 freq2=freq2(:,ok_elec);
 freq3=freq3(:,ok_elec);
 freq4=freq4(:,ok_elec);
 
 freq=[freq1;freq2;freq3;freq4];

 %%%% data stats
mem=[repmat({'hit'},size(hitind1));repmat({'miss'},size(missind1));repmat({'hit'},size(hitind2));repmat({'miss'},size(missind2))];
cond=[repmat({'word'},size([hitind1;missind1]));repmat({'face'},size([hitind2;missind2]))];
  
    for e=1:size(freq1,2)

        [p_tmp,tbl,stats] = anovan(freq(:,e),{mem cond},'model','interaction','varnames',{'mem','cond'},'display','off');               
        h(e,:)=double(p_tmp'<alpha_level);      
       % 1:mem, 2:cond, 3:interaction
           
    end
     direction(:,1)=sign((((nanmean(freq1)+nanmean(freq3)).*0.5)-((nanmean(freq2)+nanmean(freq4)).*0.5)));
     direction(:,2)=sign((((nanmean(freq1)+nanmean(freq2)).*0.5)-((nanmean(freq3)+nanmean(freq4)).*0.5)));
     direction(:,3)=sign(((nanmean(freq1)-nanmean(freq2))-(nanmean(freq3)-nanmean(freq4))));
     pos_elec_sum(n,:)=nansum((direction>0).*h);
     neg_elec_sum(n,:)=nansum((direction<0).*h);    
     ecount(n)= sum(~isnan(h(:,1)));
     
     clear direction h p_tmp tbl stats hitind1 hitind2 missind1 missind2
%%%% rand stats     
     for r=1:nrand
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
            % 1:mem, 2:cond, 3:interaction
        end
     pos_elec_randsum(n,r,:)=nansum((direction>0).*h);
     neg_elec_randsum(n,r,:)=nansum((direction<0).*h);
     clear direction h p_tmp tbl stats hitind1 hitind2 missind1 missind2
     end
clear mem mem_rand cond cond_rand freq1 freq2 freq3 freq4 freq data1 data2 f1 f2 t1 t2 r 
end
   
% calculate FFX

sum_allpat_pos_rand=squeeze(sum(pos_elec_randsum));
sum_allpat_pos=squeeze(sum(pos_elec_sum));
sum_allpat_neg_rand=squeeze(sum(neg_elec_randsum));
sum_allpat_neg=squeeze(sum(neg_elec_sum));

for con=1:3
   sorted_vec= sort(sum_allpat_pos_rand(:,con));
    test =find(sorted_vec>=sum_allpat_pos(con));
    if isempty(test)
        p_pos_ffx(con)=1./nrand;
    else
        p_pos_ffx(con)=1-(test(1)./nrand);
    end
    sorted_vec= sort(sum_allpat_neg_rand(:,con));
    test =find(sorted_vec>=sum_allpat_neg(con));
    if isempty(test)
        p_neg_ffx(con)=1./nrand;
    else
        p_neg_ffx(con)=1-(test(1)./nrand);
    end    
end

permstat.n_rand=nrand;
permstat.time=toi;
permstat.freq=foi;
permstat.neg_elec_randsum=neg_elec_randsum;
permstat.neg_elec_sum=neg_elec_sum;
permstat.p_neg_ffx=p_neg_ffx;
permstat.sum_allpat_neg=sum_allpat_neg;
permstat.sum_allpat_neg_rand=sum_allpat_neg_rand;
permstat.pos_elec_randsum=pos_elec_randsum;
permstat.pos_elec_sum=pos_elec_sum;
permstat.p_pos_ffx=p_pos_ffx;
permstat.sum_allpat_pos=sum_allpat_pos;
permstat.sum_allpat_pos_rand=sum_allpat_pos_rand;
% save stats
freqstr=strcat(num2str(foi(1)),'to',num2str(foi(2)),'Hz');
timestr=strcat(num2str(toi(1).*1000),'to',num2str(toi(2).*1000),'ms'); 
save(fullfile(path_out,strcat('permstats',sel_freq_band,freqstr,timestr)),'permstat')

fig=figure 
    subplot(1,3,1)
    histogram(permstat.sum_allpat_neg_rand(:,1),0.5:1:60)
     hold on
     plot([permstat.sum_allpat_neg(1),permstat.sum_allpat_neg(1)],[0 nrand./10],'r')
        title(strcat(effects{1},':',sel_freq_band))
    % add pvalue
    text(permstat.sum_allpat_neg(1),800,strcat('p=', num2str(permstat.p_neg_ffx(1))))
    
    subplot(1,3,2)
    histogram(permstat.sum_allpat_pos_rand(:,1),0.5:1:60)
     hold on
     plot([permstat.sum_allpat_pos(1),permstat.sum_allpat_pos(1)],[0 nrand./10],'r')
    title(strcat(effects{2},':',sel_freq_band))
    % add pvalue
    text(permstat.sum_allpat_pos(1),800,strcat('p=', num2str(permstat.p_pos_ffx(1))))    

    subplot(1,3,3)
    histogram(permstat.sum_allpat_pos_rand(:,3),0.5:1:60)
     hold on
     plot([permstat.sum_allpat_pos(3),permstat.sum_allpat_pos(3)],[0 nrand./10],'r')
    title(strcat(effects{3},':',sel_freq_band))
    % add pvalue
    text(permstat.sum_allpat_pos(3),800,strcat('p=', num2str(permstat.p_pos_ffx(3))))    
    savefig(fig,fullfile(path_figs,strcat(fig_numbers,'.fig')))

end
%%


