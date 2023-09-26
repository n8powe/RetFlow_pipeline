# Run order

# Installation procedure

1) Install python dependencies. Go to directory and run [pip install -r requirements.txt]

2) Install OpenCV with GPU support (if you have a CUDA supported graphics card.)

This is the tutorial I used to install OpenCV with GPU/CUDA support. 
https://www.freecodecamp.org/news/python-requirementstxt-explained/#:~:text=Many%20projects%20rely%20on%20libraries,be%20installed%20with%20the%20file.

Follow the instructions given to install visual studio, anaconda, CMake, Git, CUDA, and cuDNN. 

Then run installOpenCVCUDA.bat which will download, build, and install opencv with CUDA. If the .bat file does not work, follow the instructions
in the link above by running them one by one in command prompt. 

3) Then make note of the path to openCV [will look like this "//OpenCV-4.5.1//x64//vc16//bin" and should be in users on the C drive.]

4) In openCVOF2.py, undistortVideo.py, and visualizeRetinalFlow.py change the path to import OpenCV to the path from above. 
Additionally, you will need to add the path to the Nvidia GPU Computing toolkit. That is found in program files on the C drive. 

# Run order

1) Open configureVariables.m in matlab. 

2) Change any relevant paths to the data you wish to analyse. This includes the video, gaze files, timestamps, and camera intrinsics json. 

3) The booleans at the top of the file change what parts of the analyses will run. 
		preprocessVideoAndGaze -- Preprocesses the video and gaze data by undistorting them based on the camera json file. 
		chopVideo              -- Creates a video segment and the corresponding gaze data based on the undistorted files.
		runOF                  -- Runs optic flow estimation and saves the corresponding optic flow as both H5 file and individual .mat files.
		runMain                -- Calculates retinal centered optic flow based on the saved optic flow files.
		visualizeOF            -- Creates a video visualizing the optic flow and gaze location. 
		makeRFVideo            -- Creates a video visualizing retinal flow overlayed on the gaze centered video. 
		startFrame             -- The first frame of the portion of the video you wish to analyze.
		endFrame               -- The last frame of the portion of the video you wish you analyze. 
		flowMethod             -- The optic flow estimation algorithm that will be used. If no CUDA use 'deepflow' and make sure opencv is install via pip command. 
		**Other parameters will be added as needed.**

4) Run configureVariables. If paths are set correctly this should start running. If you do not have CUDA installed opencv you will need to 
install opencv-contrib-python by running pip install opencv-contrib-python.

5) Once the code gets to retinal flow processing in main.m it might crash due to dependencies. If needed, install the required matlab  modules. 

6) If everything has been setup correctly, the code will output the optic flow files, retinal flow files, and videos showing optic flow and retinal flow. 



**Below here are notes**

1) Fix the opencv cuda code in opencvOF2.py. Use Dan's code as a template. 

2) Make the changes so that this pipeline can take in Pupil Core recordings too. 