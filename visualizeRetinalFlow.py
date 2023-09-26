import os
try:
    os.add_dll_directory("C://Users//natha//OpenCV-4.5.1//x64//vc16//bin")
    os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.8//bin")
    import cv2 as cv
    useGPU = True
except:
    import cv2 as cv


import numpy as np
import time
import matplotlib.pyplot as plt

import sys
import multiprocessing
from itertools import product
from scipy.io import savemat
import scipy.io
import writeFlowFile
import h5py




def createFlowVideo(flowFolder, saveLocation, videoFile, ret_res, startFrame):

    flowFiles = os.listdir(flowFolder)

    cap = cv.VideoCapture(videoFile)
    ret, frame1 = cap.read()
    prvs = cv.cvtColor(frame1, cv.COLOR_BGR2GRAY)
    hsv = np.zeros([ret_res, ret_res, 3])
    hsv[..., 1] = 255

    blankFile = np.zeros([ret_res, ret_res, 2])

    length = int(cap.get(cv.CAP_PROP_FRAME_COUNT))

    currIndex = 0

    while(currIndex < length):
        print (currIndex)
        ret, frame2 = cap.read()

        flow = findCurrFlowFrame(startFrame+currIndex, flowFiles, blankFile, startFrame)

        flow = flow.reshape((ret_res,ret_res,2))

        #print (flow[..., 0].shape)
        #print (flow[..., 1].shape)
        mag, ang = cv.cartToPolar(flow[..., 0], flow[..., 1])
        print (hsv[:,:, 0].shape)
        hsv[:,:, 0] = ang*180/np.pi/2
        hsv[:,:, 2] = cv.normalize(mag, None, 0, 255, cv.NORM_MINMAX)
        hsv = hsv.astype(np.float32)
        #print (type(hsv))
        bgr = cv.cvtColor(hsv, cv.COLOR_HSV2BGR)
        cv.imshow('frame2', bgr)

        #k = cv.waitKey(30) & 0xff
        #if k == 27:
        #    break
        #elif k == ord('s'):
        #    cv.imwrite('opticalfb.png', frame2)
        #    cv.imwrite('opticalhsv.png', bgr)
        prvs = next

        currIndex = currIndex + 1

        plt.show()

    cv.destroyAllWindows()
        # Load MATLAB file with structures




    return 0


def findCurrFlowFrame(currIndex, flowFiles, blankFile, startFrame):

    numFrames = len(flowFiles)
    #print (numFrames)#size(flowFiles,2); # Number of flow frames.
    for i in range(0,numFrames):
        currFlowFileName = flowFiles[i]

        splitName = currFlowFileName.split('.')

        #print (splitName)

        flowFrameNumber = np.int32(splitName[0])

        if flowFrameNumber == currIndex:
            #A = fscanf(fileID,formatSpec,sizeA)
            #fileID = fopen(strcat('retflow_flo/', currFlowFileName),'r');
            #currFrame = fread(fileID, [1000 1000], 'double') * 255;%,'%f',[2, inf]);
            #close(fileID);

            #currFrame = load(strcat('retflow_flo/', currFlowFileName));
            currData = 'retflow_flo/' + currFlowFileName
            currFrame = scipy.io.loadmat(currData, struct_as_record=True)



            currFrame = np.array([currFrame['Vx'], currFrame['Vy']])

            #currFrame = h5read(strcat('retflow_flo/', currFlowFileName));
            break

        else:

            currFrame = blankFile;


    return currFrame






print('Num args:', len(sys.argv))

# Where we saved the Optic Flow files
export_dir = sys.argv[1]
# Where we save the video
filename = sys.argv[2]

videoFile = sys.argv[3]

ret_res = sys.argv[4]

startFrame = np.int32(sys.argv[5])


createFlowVideo(export_dir, filename, videoFile, np.int32(ret_res), startFrame)
