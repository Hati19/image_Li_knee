% For Creating TESTING Data set



clear all;

save_training_files='test/';
mkdir(save_training_files);


working_directory=pwd;

%give the  file name for creating testing data
mydir = '9309170-9496443/';

%parse through file for each candidate folder
d = dir(mydir);
filenames = {d(~[d.isdir]).name};
filenames = strcat(mydir, filesep, filenames); 

dirnames = {d([d.isdir]).name};
dirnames = setdiff(dirnames, {'.', '..'});
for i=1:numel(dirnames)
%for i=1:4
      fulldirname1 = [mydir  dirnames{i}]; 
      d1 = dir(fulldirname1);
      dirnames1 = {d1([d1.isdir]).name};
      dirnames1 = setdiff(dirnames1, {'.', '..'});
      for j=1:numel(dirnames1)
          fulldirname2 = [fulldirname1 filesep dirnames1{j} filesep] 
          %call function for processing single candidate
          single_candidate(fulldirname2, dirnames{i},dirnames1{j},save_training_files) 
      end
            
end



function single_candidate(filetoprocess,candidatename,V_number,save_training_files)




%read DICOM zip files and uncompress it
dinfo = dir([filetoprocess '*.zip']);
unzip([filetoprocess dinfo(1).name], filetoprocess);
%fetch the dirctory name of the DICOM files location
mydir1 = filetoprocess;
mydir1 = getdirdicom(mydir1);

%read the .mat file
dinfo = dir([filetoprocess '*.mat']);
data=load([filetoprocess dinfo.name]);

%for loop for all slices of the DICOM image to save the segment and
%training image together
for i=1:length(data.datastruct)
%for i=1:20
    if (numel(data.datastruct(i).FemoralCartilage) > 0  ) % check if FemoralCartilage exists
        % read dicom image
       img=dicomread(strcat(mydir1 , num2str(i,'%03.f')));  
       % range its values between 0 and 255
       min1=min(min(img));
       max1=max(max(img));
       img1 = uint8(255 .* ((double(img)-double(min1))) ./ double(max1-min1));
       %save DICOM image slice as .tif
       imwrite(img1,strcat(save_training_files,candidatename,'_',V_number,'_',num2str(i,'%03.f'),'.tif'));

       % read corrosponding mask from dat file


           rc=data.datastruct(i).FemoralCartilage{1};
           BW = roipoly(img,rc(:,1),rc(:,2));

       %loop for multiple segmentation in a single slice 
       for j = 2:numel(data.datastruct(i).FemoralCartilage)    

            rc=data.datastruct(i).FemoralCartilage{j};
            BW1 = roipoly(img,rc(:,1),rc(:,2));
            BW=BW1+BW;

       end
       
       imwrite(uint8(uint8(flip(BW))*255),strcat(save_training_files,candidatename,'_',V_number,'_',num2str(i,'%03.f_mask'),'.tif'));
    else
       img=dicomread(strcat(mydir1 , num2str(i,'%03.f')));  
       % range its values between 0 and 255
       min1=min(min(img));
       max1=max(max(img));
       img1 = uint8(255 .* ((double(img)-double(min1))) ./ double(max1-min1));
       %save DICOM image slice as .tif
       imwrite(img1,strcat(save_training_files,candidatename,'_',V_number,'_',num2str(i,'%03.f'),'.tif'));
       % create blank mask image
       BW  = zeros(size(img),'uint8');
       imwrite(BW,strcat(save_training_files,candidatename,'_',V_number,'_',num2str(i,'%03.f_mask'),'.tif'));
   
   end
end
%cd(working_directory)
end




function mydir = getdirdicom(mydir)
    d = dir(mydir);
    filenames = {d(~[d.isdir]).name};
    filenames = strcat(mydir, filesep, filenames); 

    dirnames = {d([d.isdir]).name};
    dirnames = setdiff(dirnames, {'.', '..'});
    

    if(numel(dirnames)~=0) 
        fulldirname = [mydir dirnames{1} filesep ]; % dirname{1} is used as only subfolder exists in current archtechture
        mydir = getdirdicom(fulldirname);       
    end
end