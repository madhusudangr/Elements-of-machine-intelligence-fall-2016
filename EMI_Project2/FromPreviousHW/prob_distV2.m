function [ genuine_scores, SimilarityMatrix,count,imposter_scores ] = prob_distV2( weights_gallery, weights_probe, number_of_coeff , distMeasure )
%PROB_DIST Summary of this function goes here
%   Detailed explanation goes here

if(strcmp(distMeasure,'euclidean'))
   SimilarityMatrix = pdist2(weights_probe',weights_gallery','euclidean'); 
elseif(strcmp(distMeasure,'chebychev'))
   SimilarityMatrix = pdist2(weights_probe',weights_gallery','chebychev'); 
elseif(strcmp(distMeasure,'minkowski'))
    SimilarityMatrix = pdist2(weights_probe',weights_gallery','minkowski');
elseif(strcmp(distMeasure,'cityblock'))
    SimilarityMatrix = pdist2(weights_probe',weights_gallery','cityblock');
end

[range,g_pdf ,i_pdf] = gen_and_imp_dist(weights_gallery, weights_probe,SimilarityMatrix);
s = ['for ' num2str(number_of_coeff) ' coefficients'];
figure('Name',s),
subplot(1,3,1), plot(range,g_pdf , range,i_pdf);








% %%%% genuine and imposter distribution %%%%%%%%%%%%
%    genuine_scores = zeros(200,1);
%    imposter_scores = zeros (200*100 - 200,1);
%    %finding genuine scores
%    k=1;
%    l =1;
%    for p = 1:1:100
%        for g = 2:2:200
% 
%            if (g == (p*2))
%                genuine_scores(k) = SimilarityMatrix(p,g-1);
%                k=k+1;
%                genuine_scores(k) = SimilarityMatrix(p,g);
%                k=k+1;
%            else
%                imposter_scores(l) = SimilarityMatrix(p,g-1);
%                l=l+1;
%                imposter_scores(l) = SimilarityMatrix(p,g);
%                l=l+1;
%            end
%            
%        end
%    end
%    max_value = max(max(genuine_scores),max(imposter_scores));
%    min_value = min(min(genuine_scores),min(imposter_scores));
%    range = min_value:30:max_value;
%    h_g = histc(genuine_scores,range);
%    g_pdf = h_g./200;
%    h_i = histc(imposter_scores,range);
%    i_pdf = h_i./19800;
%    
%     s = ['for ' num2str(number_of_coeff) ' coefficients'];
%     figure('Name',s),
%     %subplot(1,3,1), plot(range,g_pdf , range,i_pdf);
%     
% if(strcmp(distMeasure,'euclidean'))
%    subplot(1,12,1), plot(range,g_pdf , range,i_pdf); 
% elseif(strcmp(distMeasure,'chebychev'))
%    subplot(1,12,4), plot(range,g_pdf , range,i_pdf); 
% elseif(strcmp(distMeasure,'minkowski'))
%     subplot(1,12,7), plot(range,g_pdf , range,i_pdf);
% elseif(strcmp(distMeasure,'cityblock'))
%     subplot(1,12,10), plot(range,g_pdf , range,i_pdf);
% end    
%     
%     
%     
%     
%     
%     
%     
%    xlabel('Score');
%    ylabel('Probablity');
%     if(strcmp(distMeasure,'euclidean'))
%     title('Genuine and Imposter Distribution - Euclidean Dist'); 
%     elseif(strcmp(distMeasure,'chebychev'))
%     title('Genuine and Imposter Distribution - chebychev'); 
%     elseif(strcmp(distMeasure,'minkowski'))
%     title('Genuine and Imposter Distribution -minkowski');
%     elseif(strcmp(distMeasure,'cityblock'))
%     title('Genuine and Imposter Distribution -cityblock');
%     end
    
%%%%% CMC curve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%  
count = zeros(100,1);
k = 1;
l=1;
for i=1:1:100
    for j=1:1:200
        if (genuine_scores(l) <= SimilarityMatrix(i,j) || genuine_scores(l+1) <= SimilarityMatrix(i,j) )
            count(k) = count(k) + 1;  
        end
    end
    k = k+1;
    l = l+2;
end
count = 201-count;
% countofProbe = zeros(100,1);
% k=1;
% for i=1:2:200
%     countofProbe(k) =  max(count(i),count(i+1));
%     k=k+1;
% end
 %rankT_hist = histc(countofProbe,(1:1:100));
 rankT_hist = histc(count,(1:1:100));
 rankT_pdf = rankT_hist ./100;
 pointRank = zeros([100,1]);
 pointRank(1) = rankT_pdf(1);
 for i=2:1:100
     pointRank(i) = rankT_pdf(i)+pointRank(i-1);
 end
 
 %subplot(1,3,2), plot((1:1:100),pointRank);
 
if(strcmp(distMeasure,'euclidean'))
   subplot(1,12,2), plot((1:1:100),pointRank); 
elseif(strcmp(distMeasure,'chebychev'))
   subplot(1,12,5), plot((1:1:100),pointRank); 
elseif(strcmp(distMeasure,'minkowski'))
    subplot(1,12,8), plot((1:1:100),pointRank);
elseif(strcmp(distMeasure,'cityblock'))
    subplot(1,12,11), plot((1:1:100),pointRank);
end  
 
 
 
 
 
 
 xlabel('Rant(t)');
 xlabel('Rank(t)');
 ylabel('Rank-t Identification Percentage');
 title('CMC Curve');
if(strcmp(distMeasure,'euclidean'))
title('CMC Curve - Euclidean Dist'); 
elseif(strcmp(distMeasure,'chebychev'))
title('CMC Curve - chebychev'); 
elseif(strcmp(distMeasure,'minkowski'))
title('CMC Curve -minkowski');
elseif(strcmp(distMeasure,'cityblock'))
title('CMC Curve -cityblock');
end

%%%% ROC - Curve %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
m = max(max(genuine_scores),max(imposter_scores));
m = floor(m);
FNMR = zeros([1,m]);
for i=1:1:m
for j=1:2:200
if (genuine_scores(j)>i || genuine_scores(j+1) >i)
FNMR(i) = FNMR(i)+1;
end
end
FNMR(i) = FNMR(i)/200;
end
GMR = zeros([1,m]);
for i=1:1:m
GMR(i) = 1-FNMR(i);
end
FMR = zeros([1,m]);
for i=1:1:m
for j=1:1:19800
if imposter_scores(j)<i
FMR(i) = FMR(i)+1;
end
end
FMR(i) = FMR(i) /159600;
end
%subplot(1,3,3), plot(FMR,GMR);

if(strcmp(distMeasure,'euclidean'))
   subplot(1,12,3), plot(FMR,GMR); 
elseif(strcmp(distMeasure,'chebychev'))
   subplot(1,12,6), plot(FMR,GMR); 
elseif(strcmp(distMeasure,'minkowski'))
    subplot(1,12,9), plot(FMR,GMR);
elseif(strcmp(distMeasure,'cityblock'))
    subplot(1,12,12), plot(FMR,GMR);
end




title(' ROC Curve  ');
xlabel('False Match Rate (%)');
ylabel('Genuine Match Rate (%)');
title('ROC Curve');
if(strcmp(distMeasure,'euclidean'))
title('ROC Curve - Euclidean Dist'); 
elseif(strcmp(distMeasure,'chebychev'))
title('ROC Curve - chebychev'); 
elseif(strcmp(distMeasure,'minkowski'))
title('ROC Curve -minkowski');
elseif(strcmp(distMeasure,'cityblock'))
title('ROC Curve -cityblock');

end
end

function [range,g_pdf ,i_pdf] = gen_and_imp_dist(weights_gallery, weights_probe,SimilarityMatrix)
genuine_scores = zeros(200,1);
   imposter_scores = zeros (200*100 - 200,1);
   %finding genuine scores
   k=1;
   l =1;
   for p = 1:1:100
       for g = 2:2:200

           if (g == (p*2))
               genuine_scores(k) = SimilarityMatrix(p,g-1);
               k=k+1;
               genuine_scores(k) = SimilarityMatrix(p,g);
               k=k+1;
           else
               imposter_scores(l) = SimilarityMatrix(p,g-1);
               l=l+1;
               imposter_scores(l) = SimilarityMatrix(p,g);
               l=l+1;
           end
           
       end
   end
   max_value = max(max(genuine_scores),max(imposter_scores));
   min_value = min(min(genuine_scores),min(imposter_scores));
   range = min_value:30:max_value;
   h_g = histc(genuine_scores,range);
   g_pdf = h_g./200;
   h_i = histc(imposter_scores,range);
   i_pdf = h_i./19800;
   
    %s = ['for ' num2str(number_of_coeff) ' coefficients'];
    %figure('Name',s),
    %subplot(1,3,1), plot(range,g_pdf , range,i_pdf);


end


