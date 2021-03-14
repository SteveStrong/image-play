'''
Crane USSOCOM CUAS

General helper functions.

Biggs, Rego, Strong, Verrill 
February 2021
'''

# Imports
import os
import time


def ensure_save_path(save_path):
    '''
    Ensure that save_path exists. Create it if it doesn't. 
    If save_path exists and is not empty, append a rounded
    timestamp to save_path to ensure a unique path. 

    Args:
        save_path (str): Path being checked. 

    Returns:
        save_path_update (str): Either the original path, or 
            a path with an appended timestamp. 
    '''
    # Create non-existent path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        print(save_path, 'has been created.')
        return save_path
    
    # Create new save_path for non-empty save_path
    elif os.listdir(save_path):
        save_path_update = save_path + '_' + str(round(time.time()))
        os.makedirs(save_path_update)
        print(save_path, 'was not empty. ', save_path_update, 'was created and will be used instead.')
        return save_path_update

    # Return save_path for existing, empty directories
    elif not os.listdir(save_path):
        print(save_path, 'is empyt and will be used.')
        return save_path


def ensure_path_exists_and_is_non_empty(path):
    '''
    Ensure that path exists and is non-empty.

    Args:
        path (str): Path being checked. 

    Returns:
        bool: True if path exists and is non-empty, 
            False otherwise.
    '''
    if not os.path.exists(path):
        print(path, 'does not exist.')
        return False

    if not os.listdir(path):
        print(path, 'does not contain any files.')
        return False
    else:
        return True

def get_sorted_list_of_frames(path):
    '''
    Return a sorted list of frames files from a path. 

    Args:
     path (str): Path containing frames.
    '''
    # Ensure path exists
    if os.path.exists(path):
        # Get all files from path
        all_files_lst = os.listdir(path)
        # Keep only mp4's
        frames_lst = [f for f in all_files_lst if '.jpg' in f]
        # Sort frames_lst
        frames_lst.sort(key=lambda f: int(f.split('_')[0]))
        # Add paths
        frames_path_lst = [os.path.join(path, f) for f in frames_lst]
        return frames_path_lst
    else:
        print(path, 'does not exist.')
        return False


def get_sorted_list_of_clip_paths(path):
    '''
    Return a sorted list of mp4 files from a path. 

    Args:
     path (str): Path containing mp4 files.
    '''
    # Ensure path exists
    if os.path.exists(path):
        # Get all files from path
        all_files_lst = os.listdir(path)
        # Keep only mp4's
        clips_lst = [f for f in all_files_lst if '.mp4' in f]
        # Sort clips_lst
        clips_lst.sort(key=lambda f: int(f.split('.')[0]))
        # Add paths
        clips_path_lst = [os.path.join(path, f) for f in clips_lst]
        return clips_path_lst
    else:
        print(path, 'does not exist.')
        return False