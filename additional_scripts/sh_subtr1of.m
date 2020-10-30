% This function subtracts 1 over f in time frequency data by fitting a
% linear function and writes saves the slope for each time point;
% Pow is the output of ft_frequanalysis with cfg.output='pow' and
% cfg.keeptrials='yes'; Note that this function will only work if the
% dimord = rpt_chan_freq_time:

% As an input it needs:
% cfg.toi which specifies the time windows which you want to have corrected
% e.g. [-0.5 3.5]
% cfg.fit_type='robustfit' or 'linfit'
%cfg.freq2fit= [15 100] freq on which fit is done
% As an output the script generates the following variables:
% power = 1/f corrected power spectrum
% slp = ERP like structure with the 1/f slope for each time point

% Written by Simon Hanslmayr on 02/02/2017, extended Marie-Christn Fellner 

function [power]=sh_subtr1of(cfg,pow)
regression_type=cfg.fit_type;
switch pow.dimord
    case {'rpt_chan_freq_time','subj_chan_freq_time'}
        
        t1=nearest(pow.time,cfg.toi(1));
        t2=nearest(pow.time,cfg.toi(2));
        if isfield(cfg, 'freq2fit')
        f1=nearest(pow.freq,cfg.freq2fit(1));
        f2=nearest(pow.freq,cfg.freq2fit(2));
        else
        f1=  1;
        f2= numel(pow.freq);
        end
        pspc=pow.powspctrm(:,:,f1:f2,t1:t2);
        power.time=pow.time(t1:t2);
        
        freq=pow.freq(f1:f2);
        power.freq=pow.freq;
        power.dimord=pow.dimord;
        power.label=pow.label;
        
        logfrq=log10(freq);
        logfrq_all=log10(pow.freq);
        logpow=log10(pspc);
        logpow_all=log10(pow.powspctrm(:,:,:,t1:t2));

        X = [ones(length(logfrq),1) logfrq'];% we add the ones so that the intercept is also fitted
        X_all = [ones(length(logfrq_all),1) logfrq_all'];% we add the ones so that the intercept is also fitted

        ntrls=size(pspc,1);
        nchans=size(pspc,2);
        nts=size(pspc,4);
        power.slope=zeros(ntrls,nchans,nts);
        power.offset=zeros(ntrls,nchans,nts);
        power.powspctrm=NaN(ntrls,nchans,numel(pow.freq),nts);
        

        
        slpdo='rpt_chan_time';
       
        switch regression_type
            case 'linfit'
        for n=1:ntrls
            for ch=1:nchans
                for t=1:nts
                    tmp_ps=squeeze(logpow(n,ch,:,t));
                    idx=find(~isnan(tmp_ps));
                    tmp_ps=tmp_ps(idx);
                    % Fit a linear function to log transformed power
                    b = X(idx,:)\tmp_ps;% Believe it or not, this simple comand does a linear regression; Handy, innit?                   
                    linft = X_all*b;% give the linear fit
                    temp=squeeze(logpow_all(n,ch,:,t))-linft;
                    power.powspctrm(n,ch,:,t) = 10.^temp;
                    power.offset(n,ch,t)=b(1);
                    power.slope(n,ch,t)=b(2);

                end
            end
        end
            case 'robustfit'
          for n=1:ntrls
            for ch=1:nchans
                for t=1:nts
                    tmp_ps=squeeze(logpow(n,ch,:,t));
                    idx=find(~isnan(tmp_ps));
                    tmp_ps=tmp_ps(idx);
                    % Fit a linear function to log transformed power
                    if isempty(idx)
                    power.powspctrm(n,ch,idx,t) = NaN;
                    power.slope(n,ch,t)=NaN;
                    power.offset(n,ch,t)=NaN;

                    else
                        
                    b = robustfit( X(idx,2),tmp_ps);
                    linft = X_all*b;% give the linear fit
                    temp=squeeze(logpow_all(n,ch,:,t))-linft;
                    power.powspctrm(n,ch,:,t) = 10.^temp;
                    power.offset(n,ch,t)=b(1);
                     power.slope(n,ch,t)=b(2);
                    end
                end
            end
        end
        end
        
        case 'chan_freq_time'
            
        t1=nearest(pow.time,cfg.toi(1));
        t2=nearest(pow.time,cfg.toi(2));
        if isfield(cfg, 'freq2fit')
        f1=nearest(pow.freq,cfg.freq2fit(1));
        f2=nearest(pow.freq,cfg.freq2fit(2));
        else
        f1=  1;
        f2= numel(pow.freq);
        end
        pspc=pow.powspctrm(:,f1:f2,t1:t2);
        power.time=pow.time(t1:t2);
        
        freq=pow.freq(f1:f2);
        power.freq=pow.freq;
        power.dimord=pow.dimord;
        power.label=pow.label;
        
        logfrq=log10(freq);
        logfrq_all=log10(pow.freq);
        logpow=log10(pspc);
        logpow_all=log10(pow.powspctrm(:,:,t1:t2));

        X = [ones(length(logfrq),1) logfrq'];% we add the ones so that the intercept is also fitted
        X_all = [ones(length(logfrq_all),1) logfrq_all'];% we add the ones so that the intercept is also fitted

        
        nchans=size(pspc,1);
        nts=size(pspc,3);
        power.slope=zeros(nchans,nts);
        power.offset=zeros(nchans,nts);
        power.powspctrm=NaN(nchans,numel(pow.freq),nts);
        

        
        slpdo='rpt_chan_time';
       
        switch regression_type
            case 'linfit'
            for ch=1:nchans
                for t=1:nts
                    tmp_ps=squeeze(logpow(ch,:,t));
                    idx=find(~isnan(tmp_ps));
                    tmp_ps=tmp_ps(idx);
                    % Fit a linear function to log transformed power
                    b = X(idx,:)\tmp_ps';% Believe it or not, this simple comand does a linear regression; Handy, innit?                   
                    linft = X_all*b;% give the linear fit
                    temp=squeeze(logpow_all(ch,:,t))'-linft;
                    power.powspctrm(ch,:,t) = 10.^temp;
                    power.offset(ch,t)=b(1);
                    power.slope(ch,t)=b(2);

                
            end
        end
            case 'robustfit'
            for ch=1:nchans
                for t=1:nts
                    tmp_ps=squeeze(logpow(ch,:,t));
                    idx=find(~isnan(tmp_ps));
                    tmp_ps=tmp_ps(idx);
                    % Fit a linear function to log transformed power
                    if isempty(idx)
                    power.powspctrm(ch,idx,t) = NaN;
                    power.slope(ch,t)=NaN;
                    power.offset(ch,t)=NaN;

                    else
                        
                    b = robustfit( X(idx,2),tmp_ps);
                    linft = X_all*b;% give the linear fit
                    temp=squeeze(logpow_all(ch,:,t))'-linft;
                    power.powspctrm(ch,:,t) = 10.^temp;
                    power.offset(ch,t)=b(1);
                     power.slope(ch,t)=b(2);
                    
                end
            end
        end
        end  
            
        
    otherwise
        disp('Error: dimord must be rpt_chan_freq_time, subj_chan_freq_time, or chan_freq_time,  ');
end