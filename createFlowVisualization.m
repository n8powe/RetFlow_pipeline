function createFlowVisualization(flowpath, videopath, startFrame, endFrame, retinalFlow, videoName, addGaze, gazePath, allVariables)


%% This function creates a video of the retinal flow. 
    flowVisType = allVariables.flowVisType; % This isn't working. Need to fix. 
    ret_res = allVariables.ret_res;
    retImagesPath = allVariables.retImagesPath;
    

    flowFiles = {dir(fullfile(flowpath, "*.mat")).name};

    imageFiles = {dir(fullfile(retImagesPath, "*.png")).name};

    mkdir 'retFlowVideo'

    v = VideoReader(videopath);

    blankFrame = opticalFlow(zeros([v.Height, v.Width]),zeros([v.Height, v.Width])); % blank frames are for nonfixation frames. 

    v2 = VideoWriter(strcat('retFlowVideo/',videoName), 'MPEG-4');
    v2.Quality = 100;
    v2.FrameRate = 15;


    open(v2)

    currIndex = 1;

    numFrames = v.NumFrames;
    interval = allVariables.flowIntervalDownSample;

    f = figure(1);
    f.Position = [10 10 1920 880];

    [vidY, vidX]   = ndgrid(1:v.Height, 1:v.Width);
    IndexY = 1:interval+10:v.Width;
    IndexX = 1:interval+10:v.Height;
    scaleComp = 2.7;


    [vidYsquare, vidXsquare]   = ndgrid(1:ret_res, 1:ret_res);
    IndexYsquare = 1:interval:ret_res;
    IndexXsquare = 1:interval:ret_res;

    if addGaze
        gazeTable = readtable(gazePath);
    end

    if allVariables.addIMUVis
        imuData = readtable(allVariables.IMUfilePath);
        camPos = imu2position(imuData);
    end

    for frame=startFrame:endFrame-1



        vidFrame = read(v,currIndex);

        %if retinalFlow
        %    centerImage = findCenterSquareOfImage(vidFrame, ret_res);
        %end
        if allVariables.addIMUVis
            subplot(1,3,1)


        else
            subplot(1,2,1)
        end

        imshow(vidFrame)

        if addGaze
            
            hold on;
            plot(gazeTable.porX(frame), gazeTable.porY(frame), 'r+', 'MarkerSize', 40)
            hold off;

        end


        if allVariables.addIMUVis
            subplot(1,3,1)
        else
            subplot(1,2,2)
        end
        
        if retinalFlow
            blankFrame = opticalFlow(zeros([ret_res+1, ret_res+1]),zeros([ret_res+1, ret_res+1])); % blank frames are for nonfixation frames. 
            [currFrame, currImage] = findCurrFlowFrame(frame, flowFiles, blankFrame, imageFiles, ret_res);
            currFrame = opticalFlow(currFrame.Vx, currFrame.Vy);
            %imshow(vidFrame)
            %hold on;

            imshow(currImage);
            

            

            dx = currFrame.Vx(IndexXsquare, IndexYsquare);
            dy = currFrame.Vy(IndexXsquare, IndexYsquare);
            
            if strcmp(flowVisType, 'Color')
                hold on;
                flow = currFrame;
                hsv = visualizeRetinalFlowAsColors(flow, currImage, ret_res);
                imagesc(hsv)
                
                

                hold off;
            elseif strcmp(flowVisType, 'Quiver')
                hold on;
                quiver(vidXsquare(IndexXsquare, IndexYsquare), vidYsquare(IndexXsquare, IndexYsquare), dx*40, dy*40, ...
            0, 'red', 'Clipping', 'off', 'LineWidth',2);
                hold off;
            end

            hold on;
            xline(ret_res/2, 'b');
            hold off;

            hold on;
            yline(ret_res/2, 'b')
            hold off;

            
            %currFrame = opticalFlow(currFrame.dx(1:interval:end, 1:interval:end), currFrame.dy(1:interval_:end, 1:interval_:end));
            %dx = currFrame.dx(IndexX, IndexY);
            %dy = currFrame.dy(IndexX, IndexY);

            %quiver(vidX(IndexX, IndexY), vidY(IndexX, IndexY), dx*scaleComp, dy*scaleComp, ...
        %0, 'red', 'Clipping', 'off', 'LineWidth',1.25);
            %hold off;
        else
            imshow(vidFrame)
            hold on;
            currFrame = load(strcat(flowpath,flowFiles{currIndex}));
            %currFrame = opticalFlow(currFrame.dx(1:interval:end, 1:interval:end), currFrame.dy(1:interval_:end, 1:interval_:end));
            dx = currFrame.dx(IndexX, IndexY);
            dy = currFrame.dy(IndexX, IndexY);

            quiver(vidX(IndexX, IndexY), vidY(IndexX, IndexY), dx*scaleComp, dy*scaleComp, ...
        0, 'red', 'Clipping', 'off', 'LineWidth',1.25);
            hold off;
        end


        if allVariables.addIMUVis

            imuX = mean(camPos.x(camPos.index==frame))
            imuY = mean(camPos.y(camPos.index==frame))
            imuZ = mean(camPos.z(camPos.index==frame))
            subplot(1,3,3)

            plot(imuZ, 'bo')
            
        %% below here plots the IMU data as a function of frame; Will need to convert from accelerations to positions...I think 

        end


        drawnow;

        

        gcf = getframe(f);
        writeVideo(v2, gcf);


        currIndex = currIndex + 1;
    end


    close(v2);











end


function [currFrame, currRetImage] = findCurrFlowFrame(currIndex, flowFiles, blankFile, retImageFiles, ret_res)

    numFrames = size(flowFiles,2); % Number of flow frames. 
    for i=1:numFrames
        currFlowFileName = flowFiles{i};
        currRetImageName = retImageFiles{i};

        splitName = split(currFlowFileName, '.');

        flowFrameNumber = str2double(splitName{1});

        if flowFrameNumber == currIndex
            %A = fscanf(fileID,formatSpec,sizeA)
            %fileID = fopen(strcat('retflow_flo/', currFlowFileName),'r');
            %currFrame = fread(fileID, [1000 1000], 'double') * 255;%,'%f',[2, inf]);
            %fclose(fileID);

            currFrame = load(strcat('retflow_flo/', currFlowFileName));

            currRetImage = imread(strcat('ret_frames/', currRetImageName));

            %currFrame = h5read(strcat('retflow_flo/', currFlowFileName));
            break

        else

            currFrame = blankFile;
            currRetImage = zeros(ret_res,ret_res,3);

        end


    end

    


end

function centerImage = findCenterSquareOfImage(image, squareSize)

    W = size(image,2);
    H = size(image,1);

    halfSquare = 1+squareSize/2;


    centerImage = image((halfSquare-(H/2)):((H/2)+halfSquare-1), ((W/2)-halfSquare):((W/2)+halfSquare-3), :);


end


function hsv = visualizeRetinalFlowAsColors(flow, frame1, ret_res)
    %% I think I can save these in the mat files in Karl's code (main.m). 
    %hsv = zeros(ret_res+1,ret_res+1,3);
    hsv(:,:,1) = (flow.Orientation + pi) / (2*pi);
    hsv(:,:,2) = zeros(ret_res+1,ret_res+1)+1;
    hsv(:,:,3) = flow.Magnitude;
    hsv = hsv2rgb(hsv);

    %ang = flow.Orientation;
    %mag = flow.Magnitude;
    %hsv(:,:, 2) = ang;%*180/pi/2;
    %hsv(:,:, 3) = normalize(mag, "range")*255;


end


