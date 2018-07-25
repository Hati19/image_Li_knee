clear all;

save_training_files='train/';
mkdir(save_training_files);
%read DICOM zip files and uncompress it
dinfo = dir('*DICOM.zip');
%unzip(dinfo.name);

%fetch the dirctory name of the DICOM files location
mydir = pwd;
mydir = getdirdicom(mydir);

%read the .mat file
dinfo = dir('*.mat');
data=load(dinfo.name);

%for loop for all slices of the DICOM image to save the segment and
%training image together
for i=1:length(data.datastruct)
%for i=1:20
    if (numel(data.datastruct(i).FemoralCartilage) > 0  ) % check if FemoralCartilage exists
        % read dicom image
       img=dicomread(strcat(mydir ,'/', num2str(i,'%03.f')));  
       % range its values between 0 and 255
       min1=min(min(img));
       max1=max(max(img));
       img1 = uint8(255 .* ((double(img)-double(min1))) ./ double(max1-min1));
       %save DICOM image slice as .tif
       imwrite(img1,strcat(save_training_files,num2str(i,'%03.f'),'.tif'));

       % read corrosponding mask from dat file


           rc=data.datastruct(i).FemoralCartilage{1};
           BW = roipoly(img,rc(:,1),rc(:,2));

       %loop for multiple segmentation in a single slice 
       for j = 2:numel(data.datastruct(i).FemoralCartilage)    

            rc=data.datastruct(i).FemoralCartilage{j};
            BW1 = roipoly(img,rc(:,1),rc(:,2));
            BW=BW1+BW;

       end

       imwrite(flip(BW),strcat(save_training_files,num2str(i,'%03.f_mask'),'.tif'));
   end
end




function mydir = getdirdicom(mydir)
    d = dir(mydir);
    filenames = {d(~[d.isdir]).name};
    filenames = strcat(mydir, filesep, filenames); 

    dirnames = {d([d.isdir]).name};
    dirnames = setdiff(dirnames, {'.', '..'});
    

    if(numel(dirnames)~=0) 
        fulldirname = [mydir filesep dirnames{1}]; % dirname{1} is used as only subfolder exists in current archtechture
        mydir = getdirdicom(fulldirname);       
    end
end