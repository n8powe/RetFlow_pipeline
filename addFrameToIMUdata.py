import json
import numpy as np
import pandas as pd
import sys

def addFrameNumberToGazeFile(original_gaze_path, video_time_path):
    original_gaze_df = pd.read_csv(original_gaze_path)
    time_df = pd.read_csv(video_time_path)
    allTimeStamps = time_df["timestamp [ns]"]


    original_gaze_df["index"] = "NaN"


    for i in range(1,len(allTimeStamps+1)):
        currFrameStart = time_df["timestamp [ns]"][i-1]
        nextFrameStart = time_df["timestamp [ns]"][i]

        #print (i)#np.sum((original_gaze_df["timestamp [ns]"] >= currFrameStart) & (original_gaze_df["timestamp [ns]"] < nextFrameStart)))

        original_gaze_df["index"][(original_gaze_df["timestamp [ns]"] >= currFrameStart) & (original_gaze_df["timestamp [ns]"] < nextFrameStart)] = i

    return original_gaze_df



video_time_path = sys.argv[1]
imu_path = sys.argv[2]
generalPath = sys.argv[3]

imu_df = addFrameNumberToGazeFile(imu_path, video_time_path)
imu_df.to_csv(generalPath+'imu_with_time.csv')