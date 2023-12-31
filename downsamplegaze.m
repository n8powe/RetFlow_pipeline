%% DOWNSAMPLE AND UN-NORMALIZE GAZE DATA TO FIT SCREEN RESOLUTION
% REQUIREMENTS:
% 1. pull_index_normposx_normposy.m
% 2. gaze_positions.csv

function [porX porY] = downsamplegaze(norm_pos_x, norm_pos_y, index, height, width)

%% Smoove (....actually don't bother, pre-smoothing has no appreciable affect on the downsampled data!)

% %%%Debug plot
% plot((norm_pos_x),'k-o','MarkerSize',2,'DisplayName','raw');hold on 
% plot((smooth(norm_pos_x)),'r-o','MarkerSize',2,'DisplayName','moving window (3)');
% plot((smooth(norm_pos_x,'sgolay')),'b-o','MarkerSize',2,'DisplayName','sgolay(span 5 order 2)'); %% <<- I like this one JSM 2017-05-10
% plot((smooth(norm_pos_x,7,'sgolay',3)),'m-o','MarkerSize',2,'DisplayName','sgolay(span 5 order 4)');
% legend show
% hold off
% 
% norm_pos_x = smooth(norm_pos_x,'sgolay');
% norm_pos_y = smooth(norm_pos_x,'sgolay');


%% Downsample by taking mean of all eye positions on a given world-camera frame

for ii=0:max(index)
        
    if mod(ii, 100) == 0 
        disp(strcat('Resampling POR data:', num2str(ii),'-of-',num2str(max(index))))
    end
    
    if ~isempty(find(index==ii))
        porX(ii)=mean(norm_pos_x(index==ii));
        porY(ii)=mean(norm_pos_y(index==ii));
    end
end

porX = porX'*width;
porY = (1-porY')*height;

avgGazeData = table(porX, porY);
writetable(avgGazeData, 'averageGazeWithinFrame.csv');
% 
% porX_ds = porX;
% porY_ds = porY;
% 
% save('ds_distorted.mat','porX','porY');

end
