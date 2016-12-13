function [img] =  disp_img( image_array)
%DISP_IMG Summary of this function goes here
%   Detailed explanation goes here
location = '/Users/madhusudangovindraju/Downloads/Sprin 2016/Biometric Identification/HW4/images';
%get the number of images to be displayed
[n] = size(image_array,2);
img = zeros(50,50,n);
a= figure('Name','Eigen Faces for top 10 vectors');
for N=1:1:n
            img(:,:,N) =  transpose(reshape(image_array(:,N) , [50,50]));
            subplot(1,n,N) , imshow(img(:,:,N));
end
s= [location '/EigenFaces.fig'];
saveas(a,s,'fig');

end

