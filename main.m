function [] = main(subjIn, video_path, data_path, flow_path, ret_flow_path, img_out_path, allVariables)

%% params
% These params are very sensitive. We should determine what the correct
% ones for our setup are. (I think the degrees scale to resolution?)
% I (NP) should make these params available in the configureVariables file
% and then passed into this function. 

saccade_vel_thresh = allVariables.saccade_vel_thresh;
saccade_acc_thresh = allVariables.saccade_acc_thresh; % degree / second / second acceleration threshold
ret_res = allVariables.ret_res; % pixel resolution of retinal image space (square)

mkdir retflow_flo
mkdir ret_frames

%% paths

for subj = subjIn
    
    %sub_str = ['S' pad(num2str(subj),2,'left','0')];
    
    
    % video_path = ['C:/Users/natha/OneDrive/Desktop/outdoorProcessing/ExampleCode/FullDataSet/raw-data-export/2023-07-25/rawVid_undistorted.mp4']; % path directly to video (undistorted)
    % 
    % % pull from Rocks.mat / Woodchips.mat
    % %data_path = ['../' sub_str  '.csv'];
    % data_path = ['C:/Users/natha/OneDrive/Desktop/outdoorProcessing/ExampleCode/FullDataSet/raw-data-export/2023-07-25/gaze_undist.csv'];
    % flow_path = ['C:/Users/natha/OneDrive/Desktop/outdoorProcessing/ExampleCode/h5file.h5']; % path to .flo files
    % ret_flow_path = ['C:/Users/natha/OneDrive/Desktop/outdoorProcessing/ExampleCode/FullDataSet/raw-data-export/2023-07-25/retflow_flo/']; %path to output matlab flow files (retinal ref)
    % img_out_path = ['../' sub_str '/ret_img/']; %path to output retinal images (good for visualizations)
    
    %% load
    % vid = VideoReader(video_path); % video reader for image related outputs
    %         load(data_path,'porX','porY','resHeight','resWidth','calibDist','px2mmScale','framerate'); % load relevant variables
    
    %load(data_path,'rEye','lEye','calibDist','px2mmScale','norm_pos_x','norm_pos_y','gazeTable');
    
    gtable = readtable(data_path);

    %% Unsure if changing the index to this is correct. Check how it's used below [fixed]
    index = gtable.index; %This works for now ... Will need to update once we do fixation before averaging. [done] 
    height = allVariables.videoResolution(1); % Should make it so this isn't hard coded and read in from the video. 
    width = allVariables.videoResolution(2);
    norm_pos_x = gtable.gazeX_px_/width;
    %gtable.gazeY_px_ = height - gtable.gazeY_px_;
    norm_pos_y = gtable.gazeY_px_/height;
    norm_pos_z = gtable.gazeZ_px_;  % I also added the z location based on camera intrinsics. I need to double check this is correct. [assuming camera intrinsics are correct - done] 
    resHeight = allVariables.videoResolution(1);
    resWidth = allVariables.videoResolution(2);
    
    %% Unsure if this is the correct way to normalize. Going off what Karl M. emailed (me) Nate P. [looks correct - done]

    gaze_all = [gtable.gazeX_px_ - (resWidth/2), gtable.gazeY_px_ - (resHeight/2), gtable.gazeZ_px_];
    
    gaze_normal = normr(gaze_all);
    

    rEye.circle_3d_normal_x = gaze_normal(:,1);
    rEye.circle_3d_normal_y = gaze_normal(:,2);
    rEye.circle_3d_normal_z = gaze_normal(:,3);

    %% set up "frame 0" camera relative flow vector starting points
    calibDist = allVariables.calibDist;  % I need to understand these values (NP)
    px2mmScale = allVariables.px2mmScale;
    
    fixation_frames = findFixations(gtable.gazeX_px_,gtable.gazeY_px_,resWidth,resHeight,px2mmScale,calibDist,saccade_vel_thresh,saccade_acc_thresh,rEye);
    
    %porX = norm_pos_x;
    [porY, porX] = downsamplegaze(norm_pos_x, norm_pos_y, index, height, width); % Check to see if/how this works (NP) [done]
    
    
    %% Nothing below here was changed except for some resolution changes. 
    
    
    
    [xx,yy] = meshgrid(1:resWidth,1:resHeight); % x and y coordinates of pixels
    flowPre(:,:,1) = px2mmScale*(xx-resWidth/2); % 3D x coordinate of flow vector start point
    flowPre(:,:,2) = px2mmScale*(yy-resHeight/2);% 3D y coordinate of flow vector end point
    flowPre(:,:,3) = calibDist;                  % 3D z coordinate of flow vector end point (calibration distance)
    magPre = sqrt(flowPre(:,:,1).^2+flowPre(:,:,2).^2+flowPre(:,:,3).^2); % magnitude of each of the starting point vectors (used for later calculations)
    
    %% out of bound por fix
    
    porX = min(max(porX,1),resWidth); % keep these within the image resolution
    porY = min(max(porY,1),resHeight);
    
    %% dim fix
    if size(porX,1)>size(porX,2)
        porX = porX';
        porY = porY';
    end
    
    %% determine fixation vs non-fixation frames
    
    
    %         fixation_frames = findFixations(porX,porY,resWidth,resHeight,px2mmScale,calibDist,saccade_vel_thresh,framerate);
    
    
    %% list fixation blocks for easy iteration
    
    fixation_list = genFixationList(fixation_frames);
    
    fixation_list_search = fixation_list(:);
    
    idx = index(fixation_list_search);
    conv_fix_list = reshape(idx,size(fixation_list));
    
    conv_fix_list = conv_fix_list(abs(conv_fix_list(:,1)-conv_fix_list(:,2))~=0,:);
    
    fixation_list = conv_fix_list;
    
    %% determine start and end frames to process (pull from a struct?)
    
    % groundlook frames
    %           10079  to   15824
    
    
    
    startFrame = allVariables.startFrame; % fix these
    endFrame = allVariables.endFrame;%length(porX);
    
    first_fix = find(fixation_list(:,1)>startFrame,1,'first'); % first fixation within start:end
    last_fix = find(fixation_list(:,1)<endFrame,1,'last'); % last fixation within start:end
    
    addpath(genpath(cd));
    %         cd(flow_path);
    
    assert(length(porX)<=endFrame);
    
    
    %px2mmScale = 2.3232;
    %calibDist = 2100;
    %	save('testDat.mat','porX','porY');
    %% loop over fixations fix_itr goes from 1:num_fixations within start:end

    v = VideoReader(allVariables.choppedVideoFilePath);

    for fix_itr = first_fix:last_fix
        %	for fix_itr = 1;
        %         for fix_itr = 410:last_fix
        
        %% iterate through this fixation block
        c_fix_first = fixation_list(fix_itr,1); % video frame of start of fixation block
        c_fix_last = fixation_list(fix_itr,2); % video frame of end of fixation block
        
        %% loop over relevant video frames
        for frame_itr = c_fix_first:c_fix_last
            %	for frame_itr = 4208;
            tic
            %% read flow file for current frame
            try
                flow = h5read(flow_path,['/' num2str(frame_itr+1)]);
                flow = permute(flow,[3 2 1]);
                %       if frame_itr==4208
                %		writeFlowFile(flow,'4209.flo');
                %	end
            catch
                flow = ones(resHeight,resWidth,2);
            end
            %	camflow = opticalFlow(flow(:,:,1),flow(:,:,2));
            %% update gaze estimate using either eye tracker estimate (first frame of fixation block), or optic flow pathline (rest of fixation)
            
            
            flow = double(flow);
            
            eXPre = round(porX(frame_itr));
            eYPre = round(porY(frame_itr));
            
            eXPost = eXPre+ double(flow(eYPre,eXPre,1));
            eYPost = eYPre + double(flow(eYPre,eXPre,2));
            
            
            %save('testDat.mat','eYPre','eYPost','eXPre','eXPost');
            %% retinal flow field calculation + retinal image
            
            % convert current frame gaze vec to 3D world coordinates (camera relative,
            % with image plane centered at [0 0 calibDist];
            eXPre = px2mmScale*(eXPre - resWidth/2);
            eYPre = px2mmScale*(eYPre - resHeight/2);
            
            % 3D vec of eye direction
            gazePre = [eXPre eYPre calibDist];
            
            % normalized
            gazePre = normr(gazePre);
            
            % rotation matrix that rotates current gaze direction
            % to [0 0 1], use this transformation to put camera
            % relative directions into retinal reference frame
            rotm1 = axang2rotm(vrrotvec(gazePre,[0 0 1]));
            
            %  vectorize 3D meshgrid of start points of flow
            %  vectors (also the 3D meshgrid of pixel locations)
            fx = flowPre(:,:,1);
            fy = flowPre(:,:,2);
            fz = flowPre(:,:,3);
            
            % find intersection of line draw from each pixel /
            % start point of flow vectors through the nodal point
            % of your eye (gazePre, since it is the unit vector of
            % eye direction)
            % =================== Here is the pinhole eye assumption
            % =================== can modify to include real optics
            
            dir_arg = normr([gazePre(1)*ones(length(fx(:)),1)-fx(:),gazePre(2)*ones(length(fx(:)),1)-fy(:),gazePre(3)*ones(length(fx(:)),1)-fz(:)]);
            
            
            
            P = lineSphereIntersection(gazePre(1),gazePre(2),gazePre(3),...
                dir_arg(:,1),dir_arg(:,2),dir_arg(:,3),...
                0,0,0,1);
            
            % rotate these sphere intersections so that they sit in
            % retina relative space
            P = (rotm1*P')';
            
            % nan out the intersections that happen in positive Z
            P(P(:,3)>0,:) = nan;
            
            Pcalc = normr(P - [0 0 1]);
            
            Pcalc(Pcalc(:,1)==Pcalc(:,2),:) = nan;
            
            % great circle distances of each point from [0 0 -1]
            % to calculate, just find angle between that point (vector)
            % and the vector [0 0 -1];
            gcAngs = 2*atan2((vecnorm((Pcalc - [0 0 -1])')'),...
                (vecnorm((Pcalc + [0 0 -1])'))');
            
            % now push to flat back
            flatBackCoords = gcAngs.*normr(P(:,1:2));
            flatBackCoords = flatBackCoords/(pi/4);
            
            % get the corresponding indeces of points on the sphere
            % when projected back onto this square patch
            ret_dex = sub2ind([ret_res+1 ret_res+1],round(ret_res/2*flatBackCoords(:,1))+ret_res/2+1,round(ret_res/2*flatBackCoords(:,2))+ret_res/2+1);
            
            % ret_dex is the mapping of the row of P (sphere intersection
            % coordinates) to the index of the retinal image [ret_res+1 x
            % ret_res+1]
            ret_dex2 = ret_dex(~isnan(ret_dex)); % only index the non-nan values of ret_dex
            
            % get next frame's eye position vector
            eXPost = px2mmScale*(eXPost - resWidth/2);
            eYPost = px2mmScale*(eYPost - resHeight/2);
            
            
            gazePost = normr([eXPost eYPost calibDist]);
            
            % convert flowfield into mm/fr
            flow = flow*px2mmScale;
            
            % flowPost is calculated relative to flowPre, which stays the same
            
            flowPre_mod_X = flowPre(:,:,1);
            flowPre_mod_X = flowPre_mod_X(:);
            
            flowPre_mod_Y= flowPre(:,:,2);
            flowPre_mod_Y = flowPre_mod_Y(:);
            
            flowRead_x = flow(:,:,1);
            flowRead_x = flowRead_x(:);
            
            flowRead_y = flow(:,:,2);
            flowRead_y = flowRead_y(:);
            
            flowPost_mod_X = flowPre_mod_X + flowRead_x;
            flowPost_mod_Y = flowPre_mod_Y + flowRead_y;
            
            sizeMat = [size(flow,1),size(flow,2)];
            
            flowPost_mod_Z = calibDist*ones(sizeMat);
            
            flowPost = cat(3,reshape(flowPost_mod_X,sizeMat),reshape(flowPost_mod_Y,sizeMat),flowPost_mod_Z);
            
            %                 flowPost(:,:,1:2) = flowPre(:,:,1:2) + flow; % projections of points relative to the eye onto the calibration plane only move within that plane
            %                 flowPost(:,:,3) = calibDist;                 % endpoints of of the flow vectors thus stay at that depth.
            
            % now we must apply a rotation that puts gazePost at gazePre to all of
            % the flowPost vectors (add in motion of eye movement)
            % computes axis angle rotation given two 3D vectors
            rotVec = vrrotvec(gazePost,gazePre);
            
            % turn into rotation matrix for convenience
            rotmPost = axang2rotm(rotVec);
            
            
            flowPostX = flowPost(:,:,1);
            flowPostY = flowPost(:,:,2);
            flowPostZ = flowPost(:,:,3);
            
            flowPostX = flowPostX(:);
            flowPostY = flowPostY(:);
            flowPostZ = flowPostZ(:);
            
            flowPostVecs = [flowPostX(:) flowPostY(:) flowPostZ(:)];
            
            flowPostVecs = (rotmPost*flowPostVecs')';
            
            % so each flow vector (2-D, movement within the calibration plane)
            % has been converted into a start point (flowPre) and end point
            % (flowPost) 3-D rotation around the eye, with the eye movement's
            % rotation applied to the endpoints.
            
            %% calculate angles between flowPre and flowPost (rotation around vertical axis, horizontal axis, single axis)
            % also calculate the axis for each of the flow rotations
            
            
            
            fx = flowPostVecs(:,1);
            fy = flowPostVecs(:,2);
            fz = flowPostVecs(:,3);
            
            % find intersection of line draw from each pixel /
            % start point of flow vectors through the nodal point
            % of your eye (gazePre, since it is the unit vector of
            % eye direction)
            % =================== Here is the pinhole eye assumption
            % =================== can modify to include real optics
            
            dir_arg = normr([gazePre(1)*ones(length(fx(:)),1)-fx(:),gazePre(2)*ones(length(fx(:)),1)-fy(:),gazePre(3)*ones(length(fx(:)),1)-fz(:)]);
            
            % inputLine = createLine3d(fx(:),fy(:),fz(:),...
            %    dir_arg(:,1),dir_arg(:,2),dir_arg(:,3));
            
            
            P2 = lineSphereIntersection(gazePre(1),gazePre(2),gazePre(3),...
                dir_arg(:,1),dir_arg(:,2),dir_arg(:,3),...
                0,0,0,1);
            
            P2 = (rotm1*P2')';
            
            % rotate these sphere intersections so that they sit in
            % retina relative space
            % P2 = (rotm1*P2')';
            
            % nan out the intersections that happen in positive Z
            P2(isnan(P(:,3)),:) = nan;
            
            P2calc = normr(P2 - [0 0 1]);
            
            P2calc(P2calc(:,1)==P2calc(:,2),:) = nan;
            
            % great circle distances of each point from [0 0 -1]
            % to calculate, just find angle between that point (vector)
            % and the vector [0 0 -1];
            gcAngsP2 = 2*atan2((vecnorm((P2calc - [0 0 -1])')'),...
                (vecnorm((P2calc + [0 0 -1])'))');
            
            thetaP = atan2(P(:,2),P(:,1));
            thetaP2 = atan2(P2(:,2),P2(:,1));
            
            % special cases
            ccwCrossDex = and(and(and(thetaP2<0,thetaP>0),thetaP2<-pi/2),thetaP>pi/2);
            cwCrossDex = and(and(and(thetaP2>0,thetaP<0),thetaP<-pi/2),thetaP2>pi/2);
            
            dTheta = thetaP2-thetaP;
            dTheta(ccwCrossDex) = 2*pi-(abs(thetaP2(ccwCrossDex)) + abs(thetaP(ccwCrossDex)));
            dTheta(cwCrossDex) = -2*pi+(abs(thetaP2(cwCrossDex)) + abs(thetaP(cwCrossDex)));
            
            dRho = gcAngsP2 - gcAngs;
            
            %                  % retinal angVel (it still lives on the image plane at
            %                 % this point)
            %                 ret_xtest = nan(ret_res+1,ret_res+1);
            %
            %                 % same procedure for pixels
            %                 %         angVel = repmat(angVel(:),[2 1]);
            %                 ret_xtest(ret_dex2) = P(~isnan(ret_dex),1);
            %                 ret_xtest = fliplr(rot90(ret_xtest));
            %                   % retinal angVel (it still lives on the image plane at
            %                 % this point)
            %                 ret_ytest = nan(ret_res+1,ret_res+1);
            %
            %                 % same procedure for pixels
            %                 %         angVel = repmat(angVel(:),[2 1]);
            %                 ret_ytest(ret_dex2) = P(~isnan(ret_dex),2);
            %                 ret_ytest = fliplr(rot90(ret_ytest));
            %
            %                  ret_theta1 = nan(ret_res+1,ret_res+1);
            %                  ret_theta1(ret_dex2) = thetaP(~isnan(ret_dex));
            %                 ret_theta1 = fliplr(rot90(ret_theta1));
            %
            %                 ret_theta2 = nan(ret_res+1,ret_res+1);
            %                  ret_theta2(ret_dex2) = thetaP2(~isnan(ret_dex));
            %                 ret_theta2 = fliplr(rot90(ret_theta2));
            %
            %                  ccw = nan(ret_res+1,ret_res+1);
            %                  ccw(ret_dex2) = ccwCrossDex(~isnan(ret_dex));
            %                 ccw = fliplr(rot90(ccw));
            %% theta change (rotation around the viewing axis) ccw
            
            % retinal angVel (it still lives on the image plane at
            % this point)
            ret_dTheta = nan(ret_res+1,ret_res+1);
            
            % same procedure for pixels
            %         angVel = repmat(angVel(:),[2 1]);
            ret_dTheta(ret_dex2) = dTheta(~isnan(ret_dex));
            
            % get gap filler
            ret_img_fill =ret_dTheta;
            ret_img_fill(~isnan(ret_img_fill)) =1;
            ret_img_fill(isnan(ret_img_fill)) = 0;
            ret_img_mask = imfill(ret_img_fill,'holes');
            ret_img_gaps = xor(ret_img_mask,ret_img_fill);
            
            ret_dTheta(ret_img_gaps) = 0;
            ret_dTheta = regionfill(ret_dTheta,ret_img_gaps);
            % flip to properly align
            ret_dTheta = fliplr(rot90(ret_dTheta));
            ret_dTheta = rad2deg(ret_dTheta)*30;
            
            %% rho change (angle from fovea)
            
            % retinal angVel (it still lives on the image plane at
            % this point)
            ret_dRho = nan(ret_res+1,ret_res+1);
            
            % same procedure for pixels
            %         angVel = repmat(angVel(:),[2 1]);
            ret_dRho(ret_dex2) = dRho(~isnan(ret_dex));
            
            % get gap filler
            ret_img_fill =ret_dRho;
            ret_img_fill(~isnan(ret_img_fill)) =1;
            ret_img_fill(isnan(ret_img_fill)) = 0;
            ret_img_mask = imfill(ret_img_fill,'holes');
            ret_img_gaps = xor(ret_img_mask,ret_img_fill);
            
            ret_dRho(ret_img_gaps) = 0;
            ret_dRho = regionfill(ret_dRho,ret_img_gaps);
            % flip to properly align
            ret_dRho = fliplr(rot90(ret_dRho));
            ret_dRho = rad2deg(ret_dRho)*30;
            
            
            
            %% save out flows
            
            p_flow = opticalFlow(ret_dTheta,ret_dRho);
            p_flow=fliplr(rot90(p_flow,1));
            
            xyFlow = h5flo2xy(cat(3,p_flow.Vx,p_flow.Vy), allVariables);
            p_flow = xyFlow;
            
            
            [xx,yy] = meshgrid(linspace(-1,1,allVariables.ret_res+1),linspace(-1,1,allVariables.ret_res+1));
            [tt,rr] = cart2pol(xx,yy);
            
            rr(rr>1) = nan;
            
            this_phi = tt(:);
            this_theta = rr(:)*pi/4;
            % this_theta = asin(rr(:)/sqrt(2));
            
            base_vecs = [cos(this_phi).*sin(this_theta) sin(this_phi).*sin(this_theta) -cos(this_theta)];
            base_vecs = normr(base_vecs);
            base_vecs(base_vecs(:,1)==base_vecs(:,2),:) = nan;
            
            % next frame
            n_xx = xx+p_flow.Vx;
            n_yy = yy+p_flow.Vy;
            
            [n_tt,n_rr] = cart2pol(n_xx,n_yy);
            
            %     n_rr(n_rr>1) = nan;
            
            this_phi = n_tt(:);
            this_theta = n_rr(:)*pi/4;
            
            next_vecs = [cos(this_phi).*sin(this_theta) sin(this_phi).*sin(this_theta) -cos(this_theta)];
            next_vecs = normr(next_vecs);
            next_vecs(next_vecs(:,1)==next_vecs(:,2),:) = nan;
            
            mag = rad2deg(2*atan2(vecnorm([base_vecs - next_vecs]'),vecnorm([base_vecs + next_vecs]'))*30);
            
            %         mag = p_flow.Magnitude(:);
            ori = p_flow.Orientation(:);
            
            vx = p_flow.Vx(:);
            vy = p_flow.Vy(:);
            
            newVecs = normr([vx vy]).*mag';
            
            xyFlow = opticalFlow(reshape(newVecs(:,1),[allVariables.ret_res+1 allVariables.ret_res+1]),reshape(newVecs(:,2),[allVariables.ret_res+1 allVariables.ret_res+1]));
            
            p_flow = xyFlow;

            flowStruct = struct;

            flowStruct.Vx = p_flow.Vx;
            flowStruct.Vy = p_flow.Vy;
            flowStruct.Orientation = p_flow.Orientation;
            flowStruct.Magnitude = p_flow.Magnitude;
            
            %writeFlowFile(cat(3,p_flow.Vx,p_flow.Vy),[ret_flow_path ...
            %    num2str(frame_itr) '.flo']);
            
            save(strcat(ret_flow_path,num2str(frame_itr),'.mat'), "-struct", "flowStruct")

            frame = read(v,frame_itr-allVariables.startFrame);
    	    %frame = imread([img_in_path num2str(frame_itr) '.bmp']);

            % vectorize R G B channels
            frameR = frame(:,:,1);
            frameG = frame(:,:,2);
            frameB = frame(:,:,3);

            % duplicate them because of something
            frameR = repmat(frameR(:),[2 1]);
            frameG = repmat(frameG(:),[2 1]);
            frameB = repmat(frameB(:),[2 1]);



            % initialize three channels of retinal image
            ret_img_R = zeros(ret_res+1,ret_res+1);
            ret_img_G = zeros(ret_res+1,ret_res+1);
            ret_img_B = zeros(ret_res+1,ret_res+1);



            % fill in retinal image with correct pixel values
            ret_img_R(ret_dex2) = frameR(~isnan(ret_dex));
            ret_img_G(ret_dex2) = frameG(~isnan(ret_dex));
            ret_img_B(ret_dex2) = frameB(~isnan(ret_dex));

            %eflip and rotate to account for upside downness
            ret_img = fliplr(rot90(uint8(cat(3,ret_img_R,ret_img_G,ret_img_B)),1));

            % fill in holes caused by undercomplete sampling in
            % center for all three channels
            ret_img_fill =rgb2gray(ret_img);
            ret_img_fill(ret_img_fill~=0) =1;
            ret_img_mask = imfill(ret_img_fill,'holes');
            ret_img_gaps = xor(ret_img_mask,ret_img_fill);
            Rc = regionfill(ret_img(:,:,1),ret_img_gaps);
            Gc = regionfill(ret_img(:,:,2),ret_img_gaps);
            Bc = regionfill(ret_img(:,:,3),ret_img_gaps);
            ret_img = cat(3,Rc,Gc,Bc);

            % write image
            imwrite(ret_img,['ret_frames/' num2str(frame_itr) '.png']);

            %
            tt = toc;
            disp(['Progress: ' num2str(frame_itr/endFrame) ', Currently Processing '...
                num2str(frame_itr) ', ' 'Retinal Flow' ', est. time left: '...
                num2str(tt*(endFrame-frame_itr)) 'seconds']);
        end
        
    end
    
end



end
