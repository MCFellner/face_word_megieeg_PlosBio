function allstatplot_ft2019(allstat,stats_type,path_figs,fig_numbers,project_path,fieldtrip_path)
rmpath(genpath(fieldtrip_path))
addpath (fullfile(project_path,'scripts','additional_scripts','fieldtrip-20190419'));

ft_defaults
addpath (fullfile(project_path,'scripts','additional_scripts','fieldtrip-20190419','plotting'));

% reformat allstat
switch stats_type
    case 'no_cluster'
    
all_elec=[];
all_pos=[];
all_dir=[];
all_pat=[];
for n=1:numel(allstat)
all_elec=[all_elec;allstat{n}.elecs];
all_pos=[all_pos;allstat{n}.elecpos];
all_dir=[all_dir;(allstat{n}.h.*sign(allstat{n}.f))'];
all_pat=[all_pat;repmat({num2str(n)},size(allstat{n}.elecs))];
end

ok_elec=~isnan(all_dir);
all_elec=all_elec(ok_elec);
all_pos=all_pos(ok_elec,:);
all_dir=all_dir(ok_elec);
all_pat=all_pat(ok_elec);
    case 'cluster'
        
        
all_elec=allstat.all_elec;
all_pos=allstat.all_pos;
all_dir=allstat.all_dir;
all_pat=allstat.all_pat;


end
views(1,:,:)=[-90,30;90 -30;-90,0;90,0;0,-90;90 -40;];
views(2,:,:)=[90,30;-90 -30;90,0;-90,0;0,-90;-90 -40];
mesh.coordsys = 'mni';
hemispheres={'left','right'};
elec_def=[-1,1];
for h=1:numel(hemispheres)
    sel_hemi=hemispheres{h};
    sel_elec_def=elec_def(h);
    load(fullfile(project_path,'scripts','additional_scripts',strcat('surface_pial_',sel_hemi,'.mat')));
    figure
    ft_plot_mesh(mesh,'facealpha',0.2,  'edgealpha',0.2);
    hold on
    % elec to plot
    elec_toplot.unit ='mm';
    elec_toplot.coordsys ='mni';
    all_color=[1,0.5,0;0,0.5,1;0.6,0.6,0.6];%orange, blue, grey
    direction_effects={'pos','neg','none'};
    
    for d=1:numel(direction_effects)
     direction=direction_effects{d};
       none_elec=find(all_dir==0 & sign(all_pos(:,1))==sel_elec_def);

     switch direction
         % min 5 electrodes for plotting (select three none dummies,
         % overwrite in last loop step)
         case 'pos'
           sel_elec=[find(all_dir>0  & sign(all_pos(:,1))==sel_elec_def);none_elec(1:5)];
         case 'none'
           sel_elec=find(all_dir==0 & sign(all_pos(:,1))==sel_elec_def);
         case 'neg'
           sel_elec=[find(all_dir<0 & sign(all_pos(:,1))==sel_elec_def);none_elec(1:5)];
     end
     elec_check=numel(sel_elec);
    if elec_check>0
    elec_toplot.label=all_elec(sel_elec);
    elec_toplot.elecpos=all_pos(sel_elec,:);
    elec_toplot.chanpos=all_pos(sel_elec,:);
    sel_color=all_color(d,:);
    
    
    ft_plot_sens(elec_toplot,'elec','true','elecshape','sphere','facecolor',sel_color);
    else
    end
    end

    view([-90 20]);
    material dull;
    view(squeeze(views(h,1,:))');
    c1=camlight(0,0);
    set(c1, 'style', 'infinite');

    view(squeeze(views(h,2,:))');
    c2=camlight(0, 0);
    set(c2, 'style', 'infinite');

    view(squeeze(views(h,3,:))');
    print('-f1','-r600','-dtiff',fullfile(path_figs,strcat(fig_numbers,sel_hemi,'_lat.tiff'))) 

view(squeeze(views(h,4,:))');
    print('-f1','-r600','-dtiff',fullfile(path_figs,strcat(fig_numbers,sel_hemi,'_med.tiff'))) 
    clear c1 c2 

    close all
end

rmpath(genpath(fullfile(project_path,'scripts','fieldtrip-20190419')))

addpath (fieldtrip_path);
ft_defaults