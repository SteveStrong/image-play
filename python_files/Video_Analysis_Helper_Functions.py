'''
Crane USSOCOM CUAS

Video analysis helper functions. 

Biggs, Rego, Strong, Verrill 
February 2021
'''

# Imports
import os
import time
import cv2 
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import python files
import General_Helper_Functions


def create_video(frames_dir, fps):
    '''
    Create a video from saved frames.
    
    Args:
        frames_dir (str): Path containing frames.
        fps (int): Rate of frames per second.
    '''
    # Set the timer
    start_time = time.time()
    
    # Get list of frames in frames_dir
    files_lst = os.listdir(frames_dir)
    # Keep only jpg list
    frames_lst = [file for file in files_lst if file.split('.')[-1] == 'jpg']
    # Sort frames
    frames_lst.sort(key=lambda num: int(num.split('_')[-1].split('.')[0]))
    
    # Get frame width and height
    sample_frame = cv2.imread(os.path.join(frames_dir, frames_lst[0]))
    (frame_height, frame_width, _) = sample_frame.shape   
    
    # Create video object
    video_path = os.path.join(frames_dir, 'video_redo.mp4')
    save_video = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'mp4v'), fps,
                                (int(frame_width), int(frame_height)))
    
    # Iterate over frames to add them to video
    for frame in frames_lst:
        frame_path = os.path.join(frames_dir, frame)
        frame_cv2  = cv2.imread(frame_path)
        save_video.write(frame_cv2)
    
    # Close out the video
    save_video.release()
    
    # Print message
    print('Process time:', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

    
def determine_timing(start_ms, end_ms):
    '''
    Determine number of seconds between frames.

    Args:
        start_ms (int): Millisecond mark where the event begins. 
        end_ms (int): Millisecond mark where thee where the event ends.

    Returns:
        seconds (int): Number of seconds between frames. 
    '''
    # Return Flase is either start or end frame are False
    if start_ms==False or end_ms==False:
        return False
    # Return False and statement if start_frame exceeds end
    if start_ms > end_ms:
        print('Time moves only forward.')
        return False
    return (end_ms - start_ms)/1000

