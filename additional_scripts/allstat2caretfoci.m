% function to write caret foci/ focicolor files



function allstat2caretfoci(allstat,stats_type,pat,path_foci,file_name);


cd(path_foci)

header_foci='CSVF-FILE,0,,,,,,,,,,,,,,,,,,,,,,,,,\ncsvf-section-start,header,2,,,,,,,,,,,,,,,,,,,,,,,,\ntag,value,,,,,,,,,,,,,,,,,,,,,,,,,\nCaret-Version,5.65,,,,,,,,,,,,,,,,,,,,,,,,,\nDate,2014-03-17T17:01:01,,,,,,,,,,,,,,,,,,,,,,,,,\ncomment,,,,,,,,,,,,,,,,,,,,,,,,,,\nencoding,COMMA_SEPARATED_VALUE_FILE,,,,,,,,,,,,,,,,,,,,,,,,,\ncsvf-section-end,header,,,,,,,,,,,,,,,,,,,,,,,,,\ncsvf-section-start,Cells,27,,,,,,,,,,,,,,,,,,,,,,,,\nCell Number,X,Y,Z,Section,Name,Study Number,Geography,Area,Size,Statistic,Comment,Structure,Class Name,SuMS ID Number,SuMS Repeat Number,SuMS Parent Cell Base ID,SuMS Version Number,SuMS MSLID,Attribute ID,Study PubMed ID,Study Table Number,Study Table Subheader,Study Figure Number,Study Figure Panel,Study Page Reference Number,Study Page Reference Subheader';
end_foci='csvf-section-end,Cells,,,,,,,,,,,,,,,,,,,,,,,,,';
formatSpec_foci = '%d, %2.4f,%2.4f,%2.4f,,''%s'',,,,,,,,''%s'',,,,,,,,,,,,,\n';

% format of data to write: numelec, mni coord,nameeelec,pat
switch stats_type
    case 'no_cluster'
foci=cell(0,0);
nan_check=[];
for n=1:numel(pat)
   pat_vec=repmat(pat(n),numel(allstat{n}.h),1);

    numelec=num2cell(1:numel(allstat{n}.h))';
    nan_tmp=isnan(allstat{n}.h);
    mni=num2cell(allstat{n}.elecpos);
    label=strcat(allstat{n}.elecs,pat_vec);
     no_values=find(isnan(allstat{n}.f));
    foci_tmp=[mni,label,pat_vec];
    foci_tmp(no_values,:)=[];
foci=[foci;foci_tmp];
nan_check=[nan_check;nan_tmp']

end

numelec=num2cell(1:size(foci,1))';
foci=[numelec,foci];



fid=fopen(strcat('all_elecs_corr_bip.foci'),'w');

fprintf(fid,header_foci);
%fprintf(fid,formatSpec_foci,foci);
[nrows,ncols] = size(foci);
for row = 1:nrows
    fprintf(fid,formatSpec_foci,foci{row,:});
end
fprintf(fid,end_foci);
fclose(fid);


header_focicolor='CSVF-FILE,0,,,,,,,\ncsvf-section-start,header,2,,,,,,\ntag,value,,,,,,,\nCaret-Version,5.65,,,,,,,\nDate,2014-03-17T17:00:18,,,,,,,\ncomment,,,,,,,,\nencoding,COMMA_SEPARATED_VALUE_FILE,,,,,,,\npubmed_id,,,,,,,,\ncsvf-section-end,header,,,,,,,\ncsvf-section-start,Colors,9,,,,,,\nName,Red,Green,Blue,Alpha,Point-Size,Line-Size,Symbol,SuMSColorID';
end_focicolor='csvf-section-end,Colors,,,,,,,';

formatSpec_focicolor='''%s'',%d,%d,%d,255,%d,0,SPHERE,\n';

%format of data to write: nameelec, color(depending on effec),size of marker (t-value?)
foci_color=cell(0,0);
for n=1:numel(pat)
       pat_vec=repmat(pat(n),numel(allstat{n}.h),1);

    label=strcat(allstat{n}.elecs,pat_vec);
    ts=allstat{n}.f.*allstat{n}.h;
    %ts=allstat{n}.t.*allstat{n}.h.*(allstat{n}.t<0);

    color_r= num2cell(150-(100*sign(ts)));
     %  color_r=num2cell(ones(size(ts))*150);

    color_g=num2cell(ones(size(ts))*150);
     color_b= num2cell(150+(100*sign(ts)));
     %  color_b=num2cell(ones(size(ts))*150);
     %size_marker=num2cell(ones(size(allstat{n}.h))+(ones(size(allstat{n}.hcorr)).*allstat{n}.hcorr));

     size_marker=num2cell(ones(size(allstat{n}.h))+(ones(size(allstat{n}.h)).*allstat{n}.h));
    no_values=find(isnan(allstat{n}.f));
    foci_color_tmp=[label,color_r',color_g',color_b',size_marker'];
    foci_color_tmp(no_values,:)=[];

    foci_color=[foci_color;foci_color_tmp];

end

fid=fopen(strcat('all_elecs_corr_color',file_name,'.focicolor'),'w');
fprintf(fid,header_focicolor);
%fprintf(fid,formatSpec_foci,foci);
[nrows,ncols] = size(foci_color);
for row = 1:nrows
    fprintf(fid,formatSpec_focicolor,foci_color{row,:});
end
fprintf(fid,end_focicolor);
fclose(fid);


case 'cluster'
    permstats=allstat;
    alpha_cluster=0.1; % plot all clusters p<0.1

    header_focicolor='CSVF-FILE,0,,,,,,,\ncsvf-section-start,header,2,,,,,,\ntag,value,,,,,,,\nCaret-Version,5.65,,,,,,,\nDate,2014-03-17T17:00:18,,,,,,,\ncomment,,,,,,,,\nencoding,COMMA_SEPARATED_VALUE_FILE,,,,,,,\npubmed_id,,,,,,,,\ncsvf-section-end,header,,,,,,,\ncsvf-section-start,Colors,9,,,,,,\nName,Red,Green,Blue,Alpha,Point-Size,Line-Size,Symbol,SuMSColorID';
end_focicolor='csvf-section-end,Colors,,,,,,,';

formatSpec_focicolor='''%s'',%d,%d,%d,255,%d,0,SPHERE,\n';

%format of data to write: nameelec, color(depending on effec),size of marker (t-value?)
foci_color=cell(0,0);

    pat_vec=permstats.electrodes.pat;
    label=strcat(permstats.electrodes.name,pat_vec);
    
    
    % loop through different effects
    effect={'mem','cond','interaction'};
    for eff=1:numel(effect)
        
    % combine cluster info to sig neg/pos cluster vector
     % neg vector
        check_negcluster_sig=permstats.negclusterprob{eff}<alpha_cluster;
        check_negcluster_sum=permstats.negclustersum{eff}(1:numel(check_negcluster_sig))>0;
        check_negcluster=find(check_negcluster_sig&check_negcluster_sum);
        hs=zeros(numel(permstats.electrodes.name),1);
        if ~isempty(check_negcluster)
        hs=hs-(sum(permstats.negclusterlabelmat{eff}(:,check_negcluster),2));
        end
        
        % pos vector
        check_poscluster_sig=permstats.posclusterprob{eff}<alpha_cluster;
        check_poscluster_sum=permstats.posclustersum{eff}(1:numel(check_poscluster_sig))>0;
        check_poscluster=find(check_poscluster_sig&check_poscluster_sum);
        
        if ~isempty(check_poscluster)
        hs=hs+(sum(permstats.posclusterlabelmat{eff}(:,check_poscluster),2));
        end
        
        color_r= num2cell(150+(100*sign(hs)));
     %  color_r=num2cell(ones(size(ts))*150);
        color_g=num2cell(ones(size(hs))*150);
        color_b= num2cell(150-(100*sign(hs)));
     %  color_b=num2cell(ones(size(ts))*150);

     %size_marker=num2cell(ones(size(allstat{n}.h))+(ones(size(allstat{n}.hcorr)).*allstat{n}.hcorr));

     size_marker=num2cell(ones(size(hs))+(ones(size(hs)).*hs~=0));
     foci_color=[label,color_r,color_g,color_b,size_marker];

fid=fopen(strcat('all_elecs_cluster_',effect{eff},file_name,'.focicolor'),'w');
fprintf(fid,header_focicolor);
%fprintf(fid,formatSpec_foci,foci);
[nrows,ncols] = size(foci_color);
for row = 1:nrows
    fprintf(fid,formatSpec_focicolor,foci_color{row,:});
end
fprintf(fid,end_focicolor);
fclose(fid);       
    end
    
end