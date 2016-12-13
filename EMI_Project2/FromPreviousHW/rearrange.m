function [ temp ] = rearrange( im )
%REARRANGE Summary of this function goes here
%   Detailed explanation goes here

%disp('inside rearrange function');
temp = [];
%img =[];
[row,col] = size(im);
for i=1:1:row
for j=1:1:col
temp = [temp im(i,j)];
end
end
% img = [img temp];
% 
% 
% for r=2:1:size(M,3)
%     temp =[];
%     for i=1:1:20
%         for j=1:1:20
%             temp = [temp M(i,j,r)];
%         end
%     end
%     img = [img ; temp];
% end



end


