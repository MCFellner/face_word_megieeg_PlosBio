% build subfunction for clustering
% input: vec_h: logical vector,nx1 1/0 sig/non sig
%        neighboorhood mat: nxn, 1/0 define neighbouring channels
% output: num cluster
%         clustermat: rows (electrodes) x columns (1/0 for cluster def),
%         sorted, largest cluster first column

% function by Marie-Christin Fellner mcfellner@gmail.com

function [num_cluster, clustermat]=diy_neighbourcluster(vec_h,neighbourhood)
  mask_mat=tril(ones(size(neighbourhood))); % create mask mat 
  nb_nan=neighbourhood;%.*mask_mat; %mask half matrix with 0 to avoid double entries

   p_mat=(vec_h)*double(vec_h)'; % vector product significant electrodes
   p_mat_nb=p_mat.*nb_nan;% multiply sig elec mat with neighboourhood mat

  tmp=p_mat_nb(vec_h,vec_h);
  num_elec=sum(vec_h);
  ind_elec=find(vec_h);
  tmp2=zeros(num_elec);
for i=1:num_elec 
tmp2(i,:)=dot(repmat(tmp(:,i),1,num_elec),tmp);  %scalar product between column of tmp show which electrodes cluster
end
tmp2=tmp2>0;

C_tmp=1;
check=1;
while check==1
tmp3=zeros(num_elec);
for   i=1:num_elec 
  ind= find( tmp2(i,:));
  if numel(ind)>1
  tmp3(i,:)=sum(tmp2(ind,:)); %sum to clearly see cluster (check imagesc plot to clarify)
  else
  tmp3(i,:)=tmp2(i,:); % exception needed, else only one value, gets smeared across row
  end   
end
%figure; imagesc(tmp3)
%figure; imagesc(tmp2)
tmp3=tmp3>0;
[C,~,~] = unique(tmp3,'rows'); % find unique clusters
C_current=size(C,1);
check= C_tmp~=C_current;
C_tmp=C_current;
tmp2=tmp3;

end
C=C'; 
[~,ind]=sort(sum(C),'descend');

num_cluster=size(C,2);
    if num_cluster>0
    C=C(:,ind);%sort C according to cluster size
    clustermat=zeros(numel(vec_h),num_cluster);
    clustermat(ind_elec,:)=C;
    else % catch if there is not a sing significant electrode
    C=vec_h;
    num_cluster=1;
    clustermat=zeros(numel(vec_h),num_cluster);
    end

end