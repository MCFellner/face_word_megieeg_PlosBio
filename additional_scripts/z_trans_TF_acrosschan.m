% z_transformation for fieldtrip timefrequency data across channels (in
% contrast to across time -> maximizes topographical effects!)

% important: use only on all conditions concatenated! (otherwise you will
% cancel out you effects of interest)

% script by Marie-Christin Fellner mariefellner@gmx.de

function [z_freq]=z_trans_TF_acrosschan(cfg,data)
            switch data.dimord
                case 'rpt_chan_freq_time'
                    trial_dim=1;
                otherwise
                    error('fix dimension order')               
            end
            

           powspctrm=data.powspctrm;
           
           % concatenate single trials
           pow=permute(powspctrm,[2,1,3,4]);
           pow=reshape(pow,size(pow,1)*size(pow,2),size(pow,3),size(pow,4));
           
           % compute mean/std for each chan/freq
           mean_pow=squeeze(nanmean(pow));
           std_pow=squeeze(nanstd(pow));
            clear pow
           
           % reshape mean/std for matrix based calculations
           mean_pow2=repmat(reshape(mean_pow,[1,1,size(mean_pow)]),[size(powspctrm,1),size(powspctrm,2),1,1]);
           std_pow2=repmat(reshape(std_pow,[1,1,size(std_pow)]),[size(powspctrm,1),size(powspctrm,2),1,1]);

           
           % z-trans pow (pow-mean/std)
           powspctrm=(powspctrm-mean_pow2)./std_pow2;
           
           z_freq=data;
           z_freq.powspctrm=powspctrm;
           z_freq.cfg.baseline='z_trans using z_trans_TF_acrosschan';
           z_freq.cfg.z_mean=mean_pow;
           z_freq.cfg.z_std=std_pow;