'''
Crane USSOCOM CUAS

Class for iterating over frames in a video through OpenCV. 
A model can be applied to each frame for object detection. 
Object tracking may require a differentiated approach.  

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
from time import strftime, gmtime
from moviepy.video.io.VideoFileClip import VideoFileClip

# Import python files
import General_Helper_Functions        as General_Helper
import Video_Analysis_Helper_Functions as Video_Helper


# Set helper functions
def run_video(video_path, save_path, fps=False, resize=False): 
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
    opencv_vid_pr = OpenCVVideoProcessor(video_path, fps, resize, save_path)
    opencv_vid_pr.run_video()
    # Get output from model
    model_output = opencv_vid_pr.run_video()

    # Print message
    print('Process time:', time.strftime('%H:%M:%S', time.gmtime(time.time() - start_time)))

    # Return 
    return model_output


class OpenCVVideoProcessor():
    '''
    Apply pose estimation using Open CV and apply keypoint
    tracking.

    Args:
        video_path (str): Path of the video file. 
        fps (int): Rate of frames per second. If not provided, 
            every frame will be used. 
        resize (float): If provided, factor by which to rescale frames.
            E.g., 0.5 rescales at 50%.
        save_path (str): If provided, directory to save media. 
    '''
    def __init__(self, video_path, fps=False, resize=False, save_path=False):
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
        else:
            print('video_path does not exist. Class not initiated.')
            sys.exit()

    def run_video(self):
        '''
        Runs pose estimation on video as saves results. 

        Return:
            model_output (TBD): TBD
        '''
        # Read the video
        stream = cv2.VideoCapture(self.video_path)

        # Get the number of frames and the frame_rate
        total_frames = stream.get(cv2.CAP_PROP_FRAME_COUNT)
        frame_rate   = stream.get(cv2.CAP_PROP_FPS)

        # Set up object to save video
        if self.save_path:
            # Get the frame dimensions
            frame_width  = stream.get(cv2.CAP_PROP_FRAME_WIDTH)
            frame_height = stream.get(cv2.CAP_PROP_FRAME_HEIGHT)
            # Create video object
            video_path = os.path.join(self.save_path, 'video.mp4')
            if self.resize:
                save_video = cv2.VideoWriter(filename = video_path, 
                                             fourcc = cv2.VideoWriter_fourcc(*'mp4v'), 
                                             fps = self.fps,
                                             frameSize = (int(frame_width*self.resize), int(frame_height*self.resize)))
            else:
                save_video = cv2.VideoWriter(filename = video_path, 
                                             fourcc = cv2.VideoWriter_fourcc(*'mp4v'), 
                                             fps = self.fps,
                                             frameSize = (int(frame_width), int(frame_height)))

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

            # Check that frame exists as they may be returned as NoneType
            if type(frame) == np.ndarray:
                # Resize frame
                if self.resize:
                    frame = cv2.resize(frame, None, fx=self.resize, fy=self.resize)
                
                # Run frame through model
                model_output = False #self.model.apply(frame)

                # # Print the time on frame
                # text       = strftime('%H:%M:%S', gmtime(second)) + '.' + str(second).split('.')[-1]
                # font       = cv2.FONT_HERSHEY_SIMPLEX 
                # font_scale = 1
                # font_color = (0, 255, 0) 
                # thickness  = 2
                # rectangle_bgr = (0, 0, 0)
                # # get the width and height of the text box
                # (text_width, text_height) = cv2.getTextSize(text, font, font_scale, thickness)[0]
                # # set the text start position
                # text_offset_x = 10
                # text_offset_y = frame.shape[0] - 25
                # # make the coords of the box with a small padding of two pixels
                # box_coords = ((text_offset_x, text_offset_y), (text_offset_x + text_width + 2, text_offset_y - text_height - 2))
                # cv2.rectangle(frame, box_coords[0], box_coords[1], rectangle_bgr, cv2.FILLED)
                # cv2.putText(frame, text, (text_offset_x, text_offset_y), font, font_scale, font_color, thickness)

                # Save media
                if self.save_path:
                    # Save frames
                    cv2.imwrite(os.path.join(self.save_path, 
                                             str(mseconds) + '_msec.jpg'),
                                             frame)
                    # Add to video object
                    save_video.write(frame)

                # # Show image (MAY REMOVE LATER)
                # cv2.imshow('Processed Video', frame)
                # cv2.waitKey(1)

            # Kill process if frames_count exceeds total_frames_fps
            if frames_count > total_frames_fps:
                break

        # Finish the process
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
    args = vars(parser.parse_args())

    run_video(args['video_path'], 
              args['save_path'],
              args['frames_per_second'],
              args['resize'])