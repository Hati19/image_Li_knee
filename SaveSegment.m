

clear all;
img=dicomread('091');
%dinfo = dir('*DICOM.zip');
%unzip(dinfo.name);
dinfo = dir('*.mat');
data=load(dinfo.name);
rc=data.datastruct(91).FemoralCartilage{1};
BW = roipoly(img,rc(:,1),rc(:,2));
for j = 2:numel(data.datastruct(91).FemoralCartilage)    
    %rc=vertcat(rc,data.datastruct(91).FemoralCartilage{j});
    rc=data.datastruct(91).FemoralCartilage{j};
    BW1 = roipoly(img,rc(:,1),rc(:,2));
    BW=BW1+BW;

end

imwrite(flip(BW),'a.tif');