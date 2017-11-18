clear all
clc
%% calculating the maximum parcellation volume
TR=1.16;
root='/Users/yang1399/Documents/dDb/myconn/rEg_directly_filter_func';%root directory

root_mask='/Users/yang1399/Documents/dDb/mAp/ParcellationShenX';%root directory
list=dir(fullfile(root_mask,'Parcell*'));
optfile=cell(size(list));
Mmax_x=0;
Mmax_y=0;
Mmax_z=0;
mmin_x=100000;
mmin_y=100000;
mmin_z=100000;
for i=1:size(list,1)
    max_x=0;
    max_y=0;
    max_z=0;
    min_x=1000000;
    min_y=1000000;
    min_z=1000000;
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
    mask_outdir=fullfile(root,'parcellasion_timeseries','CUBE-xyz');
    if ~exist(mask_outdir,'dir')
        mkdir(mask_outdir)
    end
    % disp(strcat(datestr(datetime),' end of loading mask number_',num2str(i)))
    masksize=size(mask.img);
    % newimg=zeros(size(mask.img));
    m=0;
    for ii=1:masksize(1)
        for jj=1:masksize(2)
            for kk=1:masksize(3)
                if mask.img(ii,jj,kk)~=0
                    %% internal max-min
                    if ii > max_x
                        max_x=ii; 
                    end
                    if jj > max_y
                        max_y=jj; 
                    end
                    if kk > max_z
                        max_z=kk; 
                    end
                    if ii < min_x
                        min_x=ii; 
                    end
                    if jj < min_y
                        min_y=jj; 
                    end
                    if kk < min_z
                        min_z=kk; 
                    end
                    %% external max-min
                    if ii > Mmax_x
                        Mmax_x=ii; 
                    end
                    if jj > Mmax_y
                        Mmax_y=jj; 
                    end
                    if kk > Mmax_z
                        Mmax_z=kk; 
                    end
                    if ii < mmin_x
                        mmin_x=ii; 
                    end
                    if jj < mmin_y
                        mmin_y=jj; 
                    end
                    if kk < mmin_z
                        mmin_z=kk; 
                    end
                end
            end 
        end 
    end
    fileID = fopen(strcat(mask_outdir,'/parcellation_',num2str(i),'_maxmin_.csv'),'w');
    fprintf(fileID,'max_x= %d, min_x= %d;\n',max_x,min_x);
    fprintf(fileID,'max_y= %d, min_y= %d;\n',max_y,min_y);
    fprintf(fileID,'max_z= %d, min_z= %d;\n',max_z,min_z);
    fprintf(fileID,'length(x)= %d, width(y)= %d, deep(z)= %d;\n',max_x-min_x,max_y-min_y,max_z-min_z);
    fclose(fileID);
    disp(strcat(datestr(datetime),' done with one single mask_',num2str(i)))
end
fileID = fopen(strcat(mask_outdir,'/parcellation_','_Mmaxmmin_.txt'),'w');
fprintf(fileID,'Mmax_x= %d, mmin_x= %d;\n',Mmax_x,mmin_x);
fprintf(fileID,'Mmax_y= %d, mmin_y= %d;\n',Mmax_y,mmin_y);
fprintf(fileID,'Mmax_z= %d, mmin_z= %d;\n',Mmax_z,mmin_z);
fprintf(fileID,'length(x)= %d, width(y)= %d, deep(z)= %d;\n',Mmax_x-mmin_x,Mmax_y-mmin_y,Mmax_z-mmin_z);
fclose(fileID);
