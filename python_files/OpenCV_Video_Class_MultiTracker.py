'''
Crane USSOCOM CUAS

Class for iterating over frames in a video through OpenCV, 
with object tracking for multiple objects.
A model can be applied to each frame for object detection. 

Object tracking inspired by:
https://www.pyimagesearch.com/2018/07/30/opencv-object-tracking/

Writing to OpenCV version 4.5.1

Biggs, Rego, Strong, Verrill 
February 2021
'''

# Imports
import os
import sys
import time
import argparse
import cv2 
import numpy as np
from collections import defaultdict
from time import strftime, gmtime
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import python files
import General_Helper_Functions        as General_Helper
import Video_Analysis_Helper_Functions as Video_Helper

# Set glolbal variables
OPENCV_OBJECT_TRACKERS = {'csrt': cv2.TrackerCSRT_create, # More accurate, but slower
                          # Following are fast, but poor occlusion 
                          # performance. KCF is better
                          'kcf' : cv2.TrackerKCF_create, 
                          'mil' : cv2.TrackerMIL_create}


# Set helper functions
def run_video(video_path, save_path, fps=False, resize=False, tracker='csrt'): 
    '''
    Applies model to provided video path. 
    
    Args:
        video_path (str): Path of the video file.
        save_path (str): Directory to save media.
            Set to False if not saving. 
        fps (int): Rate of frames per second to be processed. 
            If not provided, every frame will be used. 
        resize (float): If provided, factor by which to rescale frames.
            E.g., 0.5 rescales at 50%.  
        tracker (string): OpenCV object tracker type.

    Return:
        closest_person_centroid_dict (dict): Milliseconds as key, 
            coordinates of centroid of person closest to the camera 
            as values. 
    '''
    # Set the timer
    start_time = time.time()
    
    # Ensure video_path exists 
    if not os.path.exists(video_path):
        print(video_path, 'does not exist.')
        return

    # Get video duration
    clip = VideoFileClip(video_path)
    print('Video duration:', time.strftime('%H:%M:%S', time.gmtime(clip.duration)))
    clip.reader.close()

    # Initiate and run the class
    opencv_vid_pr = OpenCVVideoProcessor(video_path, fps, resize, save_path, tracker)
    # opencv_vid_pr.run_video()
    # Get output from model
    model_output = opencv_vid_pr.run_video()

    # Print message
    print('Process time:', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

    # Return 
    return model_output


class OpenCVVideoProcessor():
    '''
    Process a video and optionally manually indicate one object to be tracked. 

    Args:
        video_path (str): Path of the video file. 
        fps (int): Rate of frames per second. If not provided, 
            every frame will be used. 
        resize (float): If provided, factor by which to rescale frames.
            E.g., 0.5 rescales at 50%.
        save_path (str): If provided, directory to save media. 
        tracker (string): OpenCV object tracker type.
    '''
    def __init__(self, video_path, fps=False, resize=False, save_path=False,
                       tracker='csrt'):
        # Initiate class if video_path exists
        if os.path.isfile(video_path):
            self.video_path = video_path
            # self.model      = TBD
            self.fps        = fps
            self.resize     = resize
            # Set save_path if provided
            if save_path:
                self.save_path = General_Helper.ensure_save_path(save_path)
            else:
                self.save_path = save_path
            self.tracker = tracker 
        else:
            print('video_path does not exist. Class not initiated.')
            sys.exit()

    def run_video(self):
        '''
        Runs video with option to manually indicate one object to be tracked. 
        While video is running, press "s" to pause the video. Use the mouse 
        to draw a bounding box around the image. Press "enter" or "space bar" 
        to continue. Press "q" to exit the loop.

        Return:
            model_output (TBD): TBD
        '''
        # Read the video
        stream = cv2.VideoCapture(self.video_path)

        # Get the number of frames, frame_rate, and dimensions 
        total_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate   = stream.get(cv2.CAP_PROP_FPS)
        frame_width  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
        frame_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)

        # Set up object to save video
        if self.save_path:
            # Create video object
            video_path = os.path.join(self.save_path, 'video.mp4')
            frame_size = (int(frame_width), int(frame_height))
            if self.resize:
                frame_size = (int(frame_width*self.resize), int(frame_height*self.resize))
            save_fps = self.fps
            if not self.fps:
                save_fps = frame_rate
            save_video = cv2.VideoWriter(filename      = video_path, 
                                         fourcc        = cv2.VideoWriter_fourcc(*'mp4v'), 
                                         fps           = save_fps,
                                         frameSize     = frame_size)

        # Calculate the total frames at the fps
        if self.fps:
            self.fps = 1 / self.fps
            total_frames_fps = round((total_frames / frame_rate) / self.fps)
        else:
            total_frames_fps = total_frames
        del total_frames

        # Set the frames count
        frames_count = 0

        # To not skip the first frame, set a flag
        first_frame_seen = False

        # Create dict to contain trackers
        tracker_dict = dict()

        # Iterate over frames in the video
        while True:
            # Use fps if set
            if self.fps:
                # Ensure first frame is not skipped
                if not first_frame_seen:
                    second = 0
                    first_frame_seen = True
                else:
                    second += self.fps
                second = round(second, 2)
                stream.set(cv2.CAP_PROP_POS_MSEC, second*1000)
            else:
                # Ensure first frame is not skipped
                if not first_frame_seen:
                    second = 0
                    first_frame_seen = True
                else:
                    second += 1/frame_rate
                second = round(second, 2)

            # Set milliseconds
            mseconds = int(second*1000)

            # Read the next frame from the file
            (grabbed, frame) = stream.read()
            frames_count     += 1 

            # Quit when the input video ends
            if not grabbed:
                break

            # Check that frame exists as they may be returned as None
            if frame is None:
                break
            
            # Resize frame
            if self.resize:
                frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize)
            
            # MODEL
            # Run frame through model
            model_output = False #self.model.apply(frame)

            # OBJECT TRACKING
            for track_bound_box, tracker in tracker_dict.items():
            # for tracker in tracker_lst:             
                # Grab the new bounding box coordinates of the object
                (track_success, box) = tracker.update(frame)

                # Draw boxes on images that been successfully tracked
                if track_success:  
                    (x, y, w, h) = [int(v) for v in box]
                    # Draw rectangle around image in frame
                    cv2.rectangle(img       = frame, 
                                  pt1       = (x, y), 
                                  pt2       = (x + w, y + h), 
                                  color     = (0, 255, 0), 
                                  thickness = 2)

            # Show image
            cv2.imshow(self.video_path, frame)
            key = cv2.waitKey(1) & 0xFF

            # TRACKER INTERACTION
            # Use mouse to select bounding box to track when 's' key is pressed
            if key == ord('s'):
                # Select the bounding box (press ENTER or SPACE to continue)
                track_bound_box = cv2.selectROI(windowName    = self.video_path, 
                                                img           = frame, 
                                                fromCenter    = False,
                                                showCrosshair = True)
                # Start OpenCV object tracker 
                tracker = OPENCV_OBJECT_TRACKERS[self.tracker]()
                tracker.init(frame, track_bound_box)
                tracker_dict[track_bound_box] = tracker

            # Use key `q` to break from the loop
            elif key == ord('q'):
                break

            # Save media
            if self.save_path:
                # Save frames
                cv2.imwrite(os.path.join(self.save_path, 
                                         str(mseconds) + '_msec.jpg'),
                                         frame)
                # Add to video object
                save_video.write(frame)

            # Kill process if frames_count exceeds total_frames_fps
            if frames_count > total_frames_fps:
                break
        
        # Release the file pointer
        stream.release()

        # Finish the process
        cv2.waitKey(0) # press any key on the keyboard to close out window
        cv2.destroyAllWindows()
        del stream
        if self.save_path:
            save_video.release()

        # Return 
        return model_output

if __name__ == '__main__':
    # Add argument parser to run commands from the command line
    parser = argparse.ArgumentParser(description='OpenCV Pose Estimation')
    parser.add_argument('-vp', '--video_path', 
                        type=str, required=True, 
                        help='Path of the video file.')  
    parser.add_argument('-sp', '--save_path', 
                        type=str, required=False, default=False,
                        help='Directory to save media. Set to False if not saving.')
    parser.add_argument('-fps', '--frames_per_second', 
                        type=int, required=False, default=False,
                        help='Rate of frames per second.')
    parser.add_argument('-rsize', '--resize', 
                        type=float, required=False, default=False,
                        help='If provided, factor by which to rescale frames.')
    parser.add_argument('-t', '--tracker', 
                        type=str, required=False, default='csrt',
	                    help='OpenCV object tracker type')
    # parser.add_argument('-show', '--show_image', 
    #                     type=str, required=False, default='csrt',
	#                     help='Use OpenCV to show image')
    args = vars(parser.parse_args())

    run_video(args['video_path'], 
              args['save_path'],
              args['frames_per_second'],
              args['resize'],
              args['tracker'])