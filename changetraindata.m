%Used to change the tring file mask name
%DO not required now
%Sibaji Gaj
%%%%%%
mydir='train/';
d = dir([mydir '*_mask.tif']);
filenames = {d(~[d.isdir]).name};
filenames = strcat(mydir, filenames); 

for i= 1: numel(filenames)
    I=imread(filenames{i});
    max(max(I))
    I1=uint8(uint8(I)*255);
    [filepath,name,ext] = fileparts(filenames{i});
    [filepath name '1' ext]
    imwrite(I1,[filepath filesep  name '1' ext])
end