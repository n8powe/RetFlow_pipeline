set cwd=%cd%	
set envName=env
set opencv-version=4.5.1
conda create -y --name %envName% numpy

cd %cwd%
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout tags/%opencv-version%
 
cd %cwd%
git clone https://github.com/opencv/opencv_contrib.git
cd opencv_contrib
git checkout tags/%opencv-version%

conda activate %envName%
set CONDA_PREFIX=%CONDA_PREFIX:\=/%
 
cd %cwd%
 
mkdir OpenCV-%opencv-version%
cd opencv
mkdir build
cd build
 
cmake ^
-G "Visual Studio 16 2019" ^
-T host=x64 ^
-DCMAKE_BUILD_TYPE=RELEASE ^
-DCMAKE_INSTALL_PREFIX=%cwd%/OpenCV-%opencv-version% ^
-DOPENCV_EXTRA_MODULES_PATH=%cwd%/opencv_contrib/modules ^
-DINSTALL_PYTHON_EXAMPLES=OFF ^
-DINSTALL_C_EXAMPLES=OFF ^
-DPYTHON_EXECUTABLE=%CONDA_PREFIX%/python3 ^
-DPYTHON3_LIBRARY=%CONDA_PREFIX%/libs/python3 ^
-DWITH_CUDA=ON ^
-DWITH_CUDNN=ON ^
-DOPENCV_DNN_CUDA=ON ^
-DWITH_CUBLAS=ON ^

cmake --build . --config Release --target INSTALL