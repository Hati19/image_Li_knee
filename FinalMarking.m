clear all;

% % I=imread('0_org.png');
% % I1=imread('0_pred.png');
% % I2=imread('0_true.png');
% % redChannel = I;
% % greenChannel = I;
% % blueChannel = I;
% % 
% % binaryImage = im2bw(I1,0.5);
% % binaryImage2 = im2bw(I2,0.5);
% % 
% % redChannel(binaryImage) = 255;
% % blueChannel(binaryImage2) = 255;
% % 
% % rgbImage = cat(3, redChannel, greenChannel, blueChannel);
% % % Display it.
% % imshow(rgbImage);
% % imwrite(rgbImage,'0_marked.png')
% for i =0:10
%     
%     I=imread([num2str(i) '_org.png']);
%     I1=imread([num2str(i) '_pred.png']);
%     I2=imread([num2str(i) '_true.png']);
%     redChannel = I;
%     greenChannel = I;
%     blueChannel = I;
% 
%     binaryImage = im2bw(I1,0.5);
%     binaryImage2 = im2bw(I2,0.5);
% 
%     redChannel(binaryImage) = 255;
%     blueChannel(binaryImage2) = 255;
% 
%     rgbImage = cat(3, redChannel, greenChannel, blueChannel);
%     % Display it.
%     imshow(rgbImage);
%     imwrite(rgbImage,[num2str(i) '_marked.png'])
% end


%give the  file name for creating marked data 
mydir = 'preds/';

%parse through file for each candidate folder
d = dir(mydir);
filenames = {d(~[d.isdir]).name};
filenames = strcat(mydir, filesep, filenames); 

dirnames = {d([d.isdir]).name};
dirnames = setdiff(dirnames, {'.', '..'});

numel(filenames)
count=0;
for i= 1: numel(filenames)
%for i=1:100
    if strfind(filenames{i}, 'org')
   % do waht you want
        %filenames{i}
        I=imread(filenames{i});
        [filepath,name,ext] = fileparts(filenames{i});
        C = strsplit(name,'_');
        filenames1=[filepath  C{1} '_' C{2} '_' C{3} '_pred' ext ]
        I1=imread([filepath  C{1} '_' C{2} '_' C{3} '_pred' ext]);
        I2=imread([filepath  C{1} '_' C{2} '_' C{3} '_true' ext]);
        redChannel = I;
        greenChannel = I;
        blueChannel = I;

        binaryImage = im2bw(I1,0.5);
        binaryImage2 = im2bw(I2,0.5);

        redChannel(binaryImage) = 255;
        blueChannel(binaryImage2) = 255;

        rgbImage = cat(3, redChannel, greenChannel, blueChannel);
        % Display it.
        imshow(rgbImage);
        imwrite(rgbImage,[filepath  C{1} '_' C{2} '_' C{3} '_marked' ext])
        count=count+1;
    end
end
count
