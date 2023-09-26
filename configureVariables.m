function allVariables = configureVariables()
    close all;
    clear all;

    clearAllData = false;
    if clearAllData
        % Haven't completed this yet. 
        clearAllDataFromWorkingDirectory();
    end

    allVariables = struct;

    'Setting parameters...'
    allVariables.preprocessVideoAndGaze = false; % This processes the entire video and can take a while. 
    allVariables.chopVideo = false;
    allVariables.runOF = false;
    allVariables.runMain = false;
    allVariables.visualizeOF = true;
    allVariables.makeRFVideo = true;
    allVariables.addIMUVis = false;

    %% Set video section to analyse
    allVariables.startFrame = 550; %% Change these values to the frames you would like to process. 
    allVariables.endFrame = 650;

    'flow method set to...[NOTE: Set to deepflow if no GPU support is enabled in opencv]' 
    %% Possible options: nvidia1, nvidia2, dense, deepflow -- nvidia2 seems best but more testing is necessary
    % There are other potential options, that I haven't implemented yet.
    % TODO in the future.
    allVariables.flowMethod = "deepflow"


    'Setting file paths...'
    allVariables.pythonCall = "python";

    allVariables.GeneralPath = "imucheck/raw-data-export/2023-09-26/";
    
    %% Separate these variables so that we know which scripts use which file paths. 
    allVariables.pythonScriptNameUndistortVideo = "undistortVideo.py";
    allVariables.pythonScriptIMUAddTime = "addFrameToIMUdata.py";
    allVariables.videoFilePath = strcat(allVariables.GeneralPath, "rawVid.mp4");
    allVariables.choppedVideoFilePath = "choppedData/choppedVideo.mp4";
    allVariables.gazeFilePath = strcat(allVariables.GeneralPath, "gaze.csv");
    allVariables.sceneCameraJsonPath = strcat(allVariables.GeneralPath, "scene_camera.json");
    allVariables.IMUfilePath = strcat(allVariables.GeneralPath, "imu.csv");
    

    allVariables.pythonScriptNameAverageGaze = "averageGazePositionPerFrame.py"; % Unused now -- leaving it until I determine if anything depends on this existing.
    allVariables.gazeFilePathUndist = "gaze_undist.csv";
    allVariables.timePath = strcat(allVariables.GeneralPath, "world_timestamps.csv");
    allVariables.retImagesPath = 'ret_frames/';



    %% Parameters for main.m
    allVariables.saccade_vel_thresh = 65; %degree / second velocity threshold for saccades
    allVariables.saccade_acc_thresh = 5; % degree / second / second acceleration threshold
    allVariables.ret_res = 1000; % pixel resolution of retinal image space (square) [I still don't fully understand this one or the next two parameters (NP)]
    allVariables.calibDist = 803; % Should this be the focal length in pixels? [Solved Yes]
    allVariables.px2mmScale = 1;

    %% Find video parameters

    vidObj = VideoReader(allVariables.videoFilePath);
    allVariables.videoResolution = [vidObj.Height, vidObj.Width];
    allVariables.flowVisType = 'Quiver'; % 'Color' doesn't work right now. Need to fix it. 
    allVariables.flowIntervalDownSample = 20;


    %% Run all scripts. 
    runRetinalFlowAllProcessing(allVariables);

    close all;
    clear all;
end

%% TODO: 
% 1) Make this code compatible with Core recordings.
% 2) Rewrite the code so that it averages the gaze positions after fixation
%       detection. [COMPLETED - IN TESTING]
% 3) Figure out how to visualize the retinal flow superimposed over the
%       video recording.
% 4) Determine how to incorporate the Zed camera output into this pipeline.
% 5) Create a .bat file that installs opencv GPU support. 
% 6) Create a requirements.txt file to install all the python dependencies.
% 


function clearAllDataFromWorkingDirectory()

    %% Make it so this function deletes the choppedVideo, Retinal Flow, and Optic Flow folders. 
    % Additionally, it should delete the average gaze position file, the
    % gaze undistorted file, the world video undistorted, and the
    % h5file.h5. 

end