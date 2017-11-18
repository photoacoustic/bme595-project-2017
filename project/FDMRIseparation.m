clear all
clc

TR=1.16;
root='/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func';%root directory
All=[{'13'}];

Nocoffee={'13','16','19','22','25','26','28','31','34','37','39','42','45','47','50','54','55','56','58','60','62','64','65','67','68','69','70','72','74','76','78','79','81','83','85','86','88','91','93','94','95','97','99','101','104'};
Coffee={'14','15','17','18','20','21','23','24','27','29','30','32','33','35','36','38','40','41','43','44','46','48','49','51','53','57','59','61','63','66','71','73','75','77','80','82','84','87','89','92','96','98','100','102','103'};
All=[Nocoffee,Coffee];

for sub=1:length(All)
imag_nii=strcat(All{sub},'_filtered_reg_data.nii.gz');

img_path=fullfile(root,imag_nii);
gz=strfind(img_path,'.gz');
if (gz~=0)
    F1=gunzip(img_path);
    image1= str2mat(F1);
    imag=load_untouch_nii(image1);
    delete(image1);
else
    imag=load_untouch_nii(img_path);
end
disp(strcat(datestr(datetime),' end of loading image nii/gz'))

root_mask='/Users/yang1399/Documents/dDb/mAp/ParcellationShenX';%root directory
list=dir(fullfile(root_mask,'Parcell*'));
optfile=cell(size(list));
max_x=0;
max_y=0;
max_z=0;
for i=1:size(list,1)
    %% load one of the mask in 278 mask
    mask_nii=list(i).name;
    mask_path=fullfile(root_mask,mask_nii);
    gz=strfind(mask_path,'.gz');
    if (gz~=0)
        F1=gunzip(mask_path);
        mask1= str2mat(F1);
        mask=load_untouch_nii(mask1);
        delete(mask1);
    else
        mask=load_untouch_nii(mask_path);
    end
    mask_outdir=fullfile(root,'parcellasion_timeseries',All{sub});
    if ~exist(mask_outdir,'dir')
        mkdir(mask_outdir)
    end
    disp(strcat(datestr(datetime),' end of loading mask number_',num2str(i)))
    masksize=size(mask.img);
    imgsize=size(imag.img);
    % newimg=zeros(size(mask.img));
    m=0;
    %fileID = fopen(strcat(mask_outdir,'/parcellation_',num2str(i),'.txt'),'w');
    csvID=fopen(strcat(mask_outdir,'/parcellation_',num2str(i),'.csv'),'w');
    for ii=1:masksize(1)
        for jj=1:masksize(2)
            for kk=1:masksize(3)
                if (mask.img(ii,jj,kk)~=0) && (imag.img(ii,jj,kk)~=0)
                    if m==0
                        disp(strcat(datestr(datetime),' start printing time siries of each mask matched voxel'))
                        %Tittle={'X','Y','Z','stD'};
                        fprintf(csvID,'X,Y,Z,stD\n');
                        %csvwrite(csvID,Tittle);
                        m=1;
                    end
                    ts=reshape(imag.img(ii,jj,kk,:),imgsize(4),1);
                    ts_filtf=filtf(demean(ts), 0.01, 0.08, 1/TR);%bandpass filter 0.01*0.08
                    %fileID = fopen(strcat(mask_outdir,'/timeseries_of_voxel_',num2str(ii),'_',num2str(jj),'_',num2str(kk),'.txt'),'w');
                    %fileID = fopen(strcat('/Users/yang1399/Documents/dDb/myconn/unprocessed/ses-',ses_tag,'/func/RS_prestats.feat/gmsYJT-detrend-filtf2012-ses-',ses_tag,'_mbsa-',num2str(sumPctsStd),'_gsa-',num2str(std(gms)),'.txt'),'w');
                    %fprintf(fileID,'%d, %d, %d, %f;\n',num2str(ii),num2str(jj),num2str(kk),std(ts_filtf));
                    fprintf(csvID,'%d,%d,%d,%4.3f\n',ii,jj,kk,std(ts_filtf));
                    %M=[ii,jj,kk,std(ts_filtf)];
                    %csvwrite(csvID,M);
                end
            end 
        end 
    end
    fclose(csvID);

    disp(strcat(datestr(datetime),' done with one single mask_',num2str(i)))
end
end

