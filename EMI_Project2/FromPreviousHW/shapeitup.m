
function [images] = shapeitup(location)

%disp('inside shapeitup function');
% for i=1:1:100
%     for j=1:1:3
%         %creating the file path
%         n = ['subject' i '_img' j '.pgm'];
%         f = fullfile(directory,'gallery_set',n);
%         %checking if the file exists
%         if exist(f, 'file') == 2
%             %if it exists read the image and reshape it to a N*1 vector
%             im = imread(f);
%             temp = rearrange(im);
%             X = [X ; temp];
%         else disp(n);disp(' not present');
%         
%     end
% end


imagelist = dir(fullfile(location,'*.pgm'));
M = size(imagelist,1);
% temp = imread(fullfile(location,imagelist(1).name));
% [row col] = size(temp);
% images = zeros(row,col,M);
% images(:,:,1) = temp;
% %load the data
% for i=2:numel(imagelist)
%     images(:,:,i) =imread(fullfile(location,imagelist(i).name));
% end

data = cell(1,numel(imagelist));
for k=1:numel(imagelist)
    data{k} = imread(fullfile(location,imagelist(k).name));
end
%images = im2double(cell2mat(images));
images = zeros(2500,M);
for i=1:1:M
    images(:,i) = rearrange(im2double(cell2mat(data(1,i))));
end

end


