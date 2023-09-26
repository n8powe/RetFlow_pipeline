function runRetinalFlowAllProcessing(allVariables)


%% Go to configureVariables to set variables paths


preprocessVideoAndGaze = allVariables.preprocessVideoAndGaze;
chopVideo = allVariables.chopVideo;
runOF = allVariables.runOF;
visualizeOF = allVariables.visualizeOF;
runMain = allVariables.runMain;
makeRFVideo = allVariables.makeRFVideo;

startFrame = allVariables.startFrame;
endFrame = allVariables.endFrame;

flowMethod = allVariables.flowMethod;


'Starting processing pipeline...'

%% Be sure all of your paths are set correctly below here. 
pythonCall = allVariables.pythonCall;
if preprocessVideoAndGaze

    'Preprocessing gaze and video...'
%% Run undistortVideo.py with correct paths to video and gaze data
    
    pythonScriptName = allVariables.pythonScriptNameUndistortVideo;
    videoFilePath = allVariables.videoFilePath;
    gazeFilePath = allVariables.gazeFilePath;
    sceneCameraJsonPath = allVariables.sceneCameraJsonPath;
    timePath = allVariables.timePath;
    arguments = [pythonCall, pythonScriptName, videoFilePath, gazeFilePath, sceneCameraJsonPath, timePath];
    system(join(arguments));

%% Run getAverageGazePositionWithinFrame.py using undistorted data. [don't use this anymore - main takes care of it]
% This function assumes gaze tracking is constantly 200Hz and downsamples it to the world video at 30Hz averaging gaze position. 
% Double check that this is correct. 
    %'Average Gaze Within Frame...'
    %pythonScriptName = allVariables.pythonScriptNameAverageGaze;
    %gazeFilePathUndist = allVariables.gazeFilePathUndist;
    %arguments_step2 = [pythonCall, pythonScriptName, gazeFilePathUndist, timePath];
    %system(join(arguments_step2));

end

if allVariables.addIMUVis
    pythonScriptName = allVariables.pythonScriptIMUAddTime;
    imuFilePath = allVariables.IMUfilePath;
    generalPath = allVariables.GeneralPath;
    timePath = allVariables.timePath;
    arguments = [pythonCall, pythonScriptName, timePath, imuFilePath, generalPath];
    system(join(arguments));

    allVariables.IMUfilePath = strcat(allVariables.GeneralPath, "imu_with_time.csv");
end

%% Run createVideoSection.m using start and end frames you wish to parse. 
%% Change these -- TODO: Make it so that it takes video time range


if chopVideo
    'Creating video segment...'
    fullVideoPath = 'rawVid_undistorted.mp4';
    
    createVideoSection(fullVideoPath, allVariables.gazeFilePathUndist, startFrame, endFrame)
end

%% Run openCVOF2.py
if runOF
    'Running optic flow estimation...'
    pythonScriptName = "openCVOF2.py";
    outputPath = "OpticFlowFiles";
    videoPath = allVariables.choppedVideoFilePath;
    maxFrames = 9999; 
    %% Add a flag for using the GPU [taken care of by the method chosen in config]
    resolution = strcat(num2str(allVariables.videoResolution(1)),'x',num2str(allVariables.videoResolution(2))); % Make sure the resolution is set to what the video was recorded at. 
    arguments = [pythonCall, pythonScriptName, outputPath, videoPath, maxFrames, resolution, flowMethod];
    %arguments = [pythonCall, "flow_source.py", videoPath]
    system(join(arguments));
end

%% Run main.m written by Karl M. Modified by Nate P. 
% There are several things to check in this that I (NP) had to adapt to the neon data. 
%% Note: Make a flag to send this that allows it to use pupil Core recordings. 
if runMain
    'Running retinal flow calculations...'
    video_path = ['choppedData/choppedVideo.mp4']; % path directly to video (undistorted)
    
    
    %data_path = ['../' sub_str  '.csv'];
    subj = 1;
    data_path = ['choppedData/choppedGazeData.csv'];
    flow_path = ['h5file.h5']; % path to .flo files
    ret_flow_path = ['retflow_flo/']; %path to output matlab flow files (retinal ref)
    sub_str = ['S' pad(num2str(subj),2,'left','0')];
    img_out_path = [sub_str '/ret_img/']; %path to output retinal images (good for visualizations)
    
    main(subj, video_path, data_path, flow_path, ret_flow_path, img_out_path, allVariables);
end

%% This is a debug step to check what the saved optic flow files look like overlayed on the world video. 
% These look good. Retinal flow below still has some issues. 
if visualizeOF
    'testing optic flow files saved...'
    opt_flow_path = ['OpticFlowFiles/choppedVideo/OpticFlow/']; %path to output matlab flow files (head centered)
    video_path = ['choppedData/choppedVideo.mp4']; % path directly to video (undistorted)
    gazeDataPath = 'averageGazeWithinFrame.csv';
    createFlowVisualization(opt_flow_path, video_path, startFrame, endFrame, false, 'opticFlowTestVideo.mp4', true, gazeDataPath, allVariables)
    close all;
end

%% add a visualization function down here that also saves the video. 
% THIS ISNT WORKING PROPERLY YET!!! Just for debugging purposes. 
%flowFromVideo();
if makeRFVideo
    'Creating flow video...'
    ret_flow_path = ['retflow_flo/']; %path to output matlab flow files (retinal ref)
    video_path = ['choppedData/choppedVideo.mp4']; % path directly to video (undistorted)
    gazeDataPath = 'averageGazeWithinFrame.csv';
    createFlowVisualization(ret_flow_path, video_path, startFrame, endFrame, true, 'retinalFlowVideo.mp4', true, gazeDataPath, allVariables)

    %    python visualizeRetinalFlow.py "retflow_flo" "choppedData/choppedVideoWithFlow.mp4" "choppedData/choppedVideo.mp4" "1201" "15650"
end


'...Finished running.'


end