function [img,weights_probe,weights_gallery,rank ] = main( )
%MAIN Summary of this function goes here
%   Detailed explanation goes here

%%%%Step 1 obtain face images, Step2:represent every image as Gamma(i)  (reshape and load the image)
location = '/Users/madhusudangovindraju/Downloads/Sprin 2016/Biometric Identification/HW4/gallery_set';
Gamma = shapeitup(location);

%%%now G will contain all M images in a 2500x1 shape and the entire matrix will be 2500*M 
% M is the number of images available
%the G is described below
%G =    | x1 x2 x3 x4 x5 ...... xM|

%where
% xi =[2500,1] of every image

%%%step 3 :find average shape vector psi (fins mean Xbar(mean))
M = size(Gamma,2);
Sumation = zeros(2500,1);
for i=1:1:M
    Sumation(:,1) = Sumation(:,1) + Gamma(:,i);
end
Psi = (Sumation)./M;
Psi = mean(Gamma,2);

%%%step 4: Subtract mean face phi(i) = Gamma(i) - psi
% put all phi(i)s together to get A
%find A 
% 
% Gamma(:,i) = [2500x1]
% 
% Psi = [2500x1]
% to calculate A is of the form [phi1 phi2 phi3 ..... phiM]
% where phi(i) is of the size [2500,1]
%
% so now A will be of the size[2500,M] %%%%%{(N^2xM)
%

%subtracting the mean face
A = zeros(2500,M);
for i=1:1:M
    A(:,i) = (Gamma(:,i)) - (Psi(:,1));  
end

%step 5: computing the covariance matrix C
%C is the covariance matrix
C = cov(A);
%C = A*transpose(A);

%step 6: compute the eigen vectors u(i)  of A*A' = C  ---> this is not
%practical - very large


%step 6.1 take the Atr_A matrix this is only MxM size and not N^2 x N^2 like
%that of C

Atr_A = transpose(A) * (A);

%step 6.2: Computer the eigen vectors of Atr_A
% get the eigen values and the eigen vectors
%eigenvalues  = mu
%eigenvectors = v
[eigenVectors,eigenvalues] = eig(Atr_A);
eig_values = diag(eigenvalues);

%step 6.3 : Compute the M best Eigen Vectors for A * Atrs
% eigen vector of A_Atrs = u
% u(i) = Av(i)

[sorted_EigVal, I] = sort(eig_values,'descend');

%%%question1 take the top 10 eigen values
top10EigenVectors = zeros(200,10);
for i=1:1:10
 top10EigenVectors(:,i) = eigenVectors( :,I(i));
end
%eigen vectors of A*Atrs is u(i) = A * v(i);
top10EigenVectors_of_AAtrs = zeros(2500,10);
top10EigenVectors_of_AAtrs = (A) * top10EigenVectors;
assignin('base', 'top10EigenVectors_of_AAtrs', top10EigenVectors_of_AAtrs);
save('demo');
img = disp_img(top10EigenVectors_of_AAtrs);

%%% Question 2 : Select the top 30 eigen-faces and compute the eigen-coefficients of all the images in the dataset

top30EigenVectors = zeros(200,30);
for i=1:1:30
 top30EigenVectors(:,i) = eigenVectors( :,I(i));
end
top30EigenVectors_of_AAtrs = (A) * top30EigenVectors;
%load the probe
location = '/Users/madhusudangovindraju/Downloads/Sprin 2016/Biometric Identification/HW4/probe_set';
Probe = shapeitup(location);
numberofProbeImgs = size(Probe,2);
probeA = zeros(2500,numberofProbeImgs);
for i=1:1:numberofProbeImgs
    probeA(:,i) = (Probe(:,i)) - (Psi(:,1));  
end
s = ['top' num2str(30) 'Eigenfaces'];
assignin('base', s, top30EigenVectors_of_AAtrs);
save('demo');
%calculate the weights
weights_probe = transpose(top30EigenVectors_of_AAtrs) * probeA;
weights_gallery = transpose(top30EigenVectors_of_AAtrs) * A;
assignin('base','wieghts_probe_30', weights_probe);
assignin('base','weights_gallery_30',weights_gallery);
save ('demo');

%%%% Q3 , Q4,  Q5, Q6, Q7
ranks = zeros(10,4); % to store the rank 1 identfication rates for different coefficients
s = ['for ' num2str(30) ' coefficients'];
fh = figure('Name',s);
[ genuine_scores, SimilarityMatrix,count,imposter_scores , Rank_Eu ] = prob_dist( weights_gallery, weights_probe , 30 , 'euclidean' );
[ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_Ch] = prob_dist( weights_gallery, weights_probe , 30 , 'chebychev' );
[ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_Min] = prob_dist( weights_gallery, weights_probe , 30 , 'minkowski' );
[ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_CB] = prob_dist( weights_gallery, weights_probe , 30 , 'cityblock' );

%%%% we are storing the ranks to find the rank 1 recognition rates for
%%%% different coefficients
rank(3,1) = Rank_Eu(1);
rank(3,2) = Rank_Ch(1);
rank(3,3) = Rank_Min(1);
rank(3,4) = Rank_CB(1);
saveas(fh,'for 30 coefficients.fig');

%%% now repeat the same for different number of coefficients from 40 to 100
disp('finished calculating for 30 coeff, noe calculating the rest');
for num = 40:10:100
    topEigenVectors = zeros(200,30);
    for i=1:1:30
     topEigenVectors(:,i) = eigenVectors( :,I(i));
    end
    topEigenVectors_of_AAtrs = (A) * topEigenVectors;
    s = ['top' num2str(num) 'Eigenfaces'];
    assignin('base', s, topEigenVectors_of_AAtrs);
    save('demo');
    %load the probe
    location = '/Users/madhusudangovindraju/Downloads/Sprin 2016/Biometric Identification/HW4/probe_set';
    Probe = shapeitup(location);
    numberofProbeImgs = size(Probe,2);
    probeA = zeros(2500,numberofProbeImgs);
    for i=1:1:numberofProbeImgs
        probeA(:,i) = (Probe(:,i)) - (Psi(:,1));  
    end
    %calculate the weights
    weights_probe = transpose(top30EigenVectors_of_AAtrs) * probeA;
    weights_gallery = transpose(top30EigenVectors_of_AAtrs) * A;
    s = ['weights_probe_' num2str(num)];
    assignin('base',s,weights_probe);
    s = ['weights_gallery_' num2str(num)];
    assignin('base',s,weights_gallery);
    save('demo');
    s = ['for ' num2str(num) ' coefficients'];
    fh = figure('Name',s);
    
    [ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_Eu ] = prob_dist( weights_gallery, weights_probe , num , 'euclidean' );
    [ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_Ch ] = prob_dist( weights_gallery, weights_probe , num , 'chebychev' );
    [ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_Min ] = prob_dist( weights_gallery, weights_probe , num , 'minkowski' );
    [ genuine_scores, SimilarityMatrix,count,imposter_scores ,Rank_CB] = prob_dist( weights_gallery, weights_probe , num , 'cityblock' );
    s = ['for ' num2str(num) ' coefficients.fig'];
    saveas(fh,s);
%%%% we are storing the ranks to find the rank 1 recognition rates for
%%%% different coefficients
    rank(num/10,1) = Rank_Eu(1);
    rank(num/10,2) = Rank_Ch(1);
    rank(num/10,3) = Rank_Min(1);
    rank(num/10,4) = Rank_CB(1);
end


%%% now to find the rank 1 recognition rates for different distance.
fh = figure('Name','Rank 1 recognition rate For Eucledian Distance');
rank = rank * 100;
r = zeros(10,1);
for num = 3:1:10
    r(num,1) = rank(num,1);
end
plot(30:10:100 , r(3:1:10));
xlabel('Coefficient');
ylabel('Rank-1 Recognition Rate');
title('Rank 1 recognition rate For Eucledian Distance');
s = 'Rank 1 recognition rate For Eucledian Distance.fig';
saveas(fh,s);


fh = figure('Name','Rank 1 recognition rate For Chebychev Distance');
r = zeros(10,1);
for num = 3:1:10
    r(num,1) = rank(num,2);
end
plot(30:10:100 , r(3:1:10));
xlabel('Coefficient');
ylabel('Rank-1 Recognition Rate');
title('Rank 1 recognition rate For Chebychev Distance');
s = 'Rank 1 recognition rate For Chebychev Distance.fig';
saveas(fh,s);


fh = figure('Name','Rank 1 recognition rate For Minkowski Distance');
r = zeros(10,1);
for num = 3:1:10
    r(num,1) = rank(num,3);
end
plot(30:10:100 , r(3:1:10));
xlabel('Coefficient');
ylabel('Rank-1 Recognition Rate');
title('Rank 1 recognition rate For Minkowski Distance');
s = 'Rank 1 recognition rate For Minkowski Distance.fig';
saveas(fh,s);


fh = figure('Name','Rank 1 recognition rate For CityBlock Distance');
r = zeros(10,1);
for num = 3:1:10
    r(num,1) = rank(num,4);
end
plot(30:10:100 , r(3:1:10));
xlabel('Coefficient');
ylabel('Rank-1 Recognition Rate');
title('Rank 1 recognition rate For CityBlock Distance');
s = 'Rank 1 recognition rate For CityBlock Distance.fig';
saveas(fh,s);





end


