import json
import numpy as np
#with open("scene_camera.json", "r") as f:
#  data = json.load(f)

#camera_matrix = np.array(data["camera_matrix"])
#dist_coeffs = np.array(data["distortion_coefficients"])

#print("Camera Matrix:")
#print(camera_matrix)
#print("Distortion Coefficients:")
#print(dist_coeffs)

import os

import pathlib

import av
from tqdm import tqdm
import pandas as pd
try:
    os.add_dll_directory("C://Users//HayhoeLab//OpenCV-4.5.1//x64//vc16//bin")
    os.add_dll_directory("C://Program Files//NVIDIA GPU Computing Toolkit//CUDA//v11.0//bin")
    import cv2
    useGPU = True
except:
    import cv2
    useGPU = False
import os
import sys

def undistort_video(
    original_video_path, undistorted_video_path, scene_camera_path
):
    timestamps_path = pathlib.Path(original_video_path).with_name(
        "world_timestamps.csv"
    )
    num_frames = pd.read_csv(timestamps_path).shape[0]
    original_container = av.open(str(original_video_path))
    original_video_stream = original_container.streams.video[0]

    undistorted_container = av.open(str(undistorted_video_path), "w")

    with open(scene_camera_path, "r") as f:
      data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    dist_coeffs = np.array(data["distortion_coefficients"])

    try:
        undistorted_video = undistorted_container.add_stream("h264_nvenc")
    except Exception as e:
        print("nvenc not available", e)
        undistorted_video = undistorted_container.add_stream("h264")

    undistorted_video.options["bf"] = "0"
    undistorted_video.options["movflags"] = "faststart"
    undistorted_video.gop_size = original_video_stream.gop_size
    undistorted_video.codec_context.height = original_video_stream.height
    undistorted_video.codec_context.width = original_video_stream.width
    undistorted_video.codec_context.time_base = original_video_stream.time_base
    undistorted_video.codec_context.bit_rate = original_video_stream.bit_rate

    if original_container.streams.audio:
        audio_stream = original_container.streams.audio[0]
        output_audio_stream = undistorted_container.add_stream("aac")
        output_audio_stream.codec_context.layout = audio_stream.layout.name
        output_audio_stream.codec_context.time_base = audio_stream.time_base
        output_audio_stream.codec_context.bit_rate = audio_stream.bit_rate
        output_audio_stream.codec_context.sample_rate = audio_stream.sample_rate

    progress = tqdm(unit=" frames", total=num_frames)
    with undistorted_container:
        for packet in original_container.demux():
            frames = packet.decode()

            if packet.stream.type == "audio":
                for frame in frames:
                    packets = output_audio_stream.encode(frame)
                    undistorted_container.mux(packets)
            elif packet.stream.type == "video":
                for frame in frames:
                    #print ("Frame---", frame)
                    img = frame.to_ndarray(format="bgr24")
                    undistorted_img = cv2.undistort(img, camera_matrix, dist_coeffs)
                    new_frame = frame.from_ndarray(undistorted_img, format="bgr24")
                    new_frame.pts = frame.pts
                    new_frame.time_base = original_video_stream.time_base
                    packets = undistorted_video.encode(new_frame)
                    progress.update()
                    undistorted_container.mux(packets)
        # encode and mux frames that have been queued internally by the encoders
        undistorted_container.mux(output_audio_stream.encode())
        undistorted_container.mux(undistorted_video.encode())


def undistort_gaze(original_gaze_path, unditorted_gaze_path, scene_camera_path, video_time_path):

    with open(scene_camera_path, "r") as f:
      data = json.load(f)

    camera_matrix = np.array(data["camera_matrix"])
    #print ("Camera matrix = ", camera_matrix)
    focalLength = camera_matrix[[0]][0,0]
    print ("Focal Length = ", focalLength)

    dist_coeffs = np.array(data["distortion_coefficients"])
    print (original_gaze_path)
    original_gaze_df = addFrameNumberToGazeFile(original_gaze_path, video_time_path)
    original_gaze = original_gaze_df[["gaze x [px]", "gaze y [px]"]].values
    undistorted_gaze = cv2.undistortPoints(
        original_gaze.reshape(-1, 2), camera_matrix, dist_coeffs, P=camera_matrix
    )

    undistorted_gaze_df = original_gaze_df.copy()
    undistorted_gaze_df[["gaze x [px]", "gaze y [px]"]] = undistorted_gaze.reshape(-1, 2)
    undistorted_gaze_df[["gaze z [px]"]] = np.ones([len(undistorted_gaze_df[["gaze x [px]"]]),1])*focalLength
    #print ("Pz = ", undistorted_gaze_df[["gaze_z_px"]])
    undistorted_gaze_df.to_csv(unditorted_gaze_path, index=False)



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



recording_folder = ""
scene_camera_path = sys.argv[3]
original_video_path = sys.argv[1]
undistorted_video_path = recording_folder + "rawVid_undistorted.mp4"
print ("Undistoring video...")
undistort_video(original_video_path, undistorted_video_path, scene_camera_path)

original_gaze_path = sys.argv[2]
video_time_path = sys.argv[4]
undistorted_gaze_path = recording_folder + "gaze_undist.csv"
print ("Undistorting Gaze Positions...")
undistort_gaze(original_gaze_path, undistorted_gaze_path, scene_camera_path, video_time_path)
