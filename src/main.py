import os
import sys
import cv2
import numpy as np

def load_video_from_argv():
     # Getting the file path as an argument:
    try:
        rel_path = sys.argv[1] # The first argument is the path (second in argv, after the script name)
        data_path = os.path.abspath(rel_path)
        print(f"The file is: {data_path}")
    except:
        print("No input given. Script finished.")
        exit() # If no file is informed, the script is finished.

    # Verifying if the given file exists:
    if not os.path.exists(data_path):
        print("The file does not exist. Script finished.")
        exit()

    # Every verification concerning the file have been performed at this point.

    return cv2.VideoCapture(data_path) # Returning the video...

def show_video(video):
    video_frames = [] # The frames are saved to be used later
    framerate = 60
    while video.isOpened():
        _, frame = video.read() # Loading each frame
        
        if type(frame) == np.ndarray:
            cv2.imshow('My video', frame) # Showing each frame of the video
            video_frames.append(frame)
        else:
            break
        if cv2.waitKey(100 // framerate) == ord('k'):
            break
        

    video.release()
    cv2.destroyAllWindows()

    return video_frames

def perform_bg_subtraction(video_frames):
    # Getting the video frames count:
    N_frames = len(video_frames)

    # Defining the MOG2:
    mog2 = cv2.createBackgroundSubtractorMOG2(N_frames)

    # Applying the images into the MOG2
    mogged_frames = []
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mog_iter = mog2.apply(img) # This works like an "average"
        mogged_frames.append(cv2.cvtColor(mog_iter, cv2.COLOR_BGR2RGB))

    # Retuning the obtained background:
    return cv2.cvtColor(mog2.getBackgroundImage(), cv2.COLOR_BGR2RGB), mogged_frames

def play_video(video_frames, framerate, res_func):
    for frame in video_frames:
        result = res_func(frame)
        cv2.imshow('My video', result)

        if cv2.waitKey(1000 // framerate) == ord('k'): # If you press 'k' the window will close
            break

    cv2.destroyAllWindows()

def treat_image(frame):
    # Performing the requested threshold treatment:
    _, thresh = cv2.threshold(frame, 127, 255, cv2.THRESH_BINARY_INV)

    # Performing the erosion and the dilatation:
    kernel_erosion = np.ones((9, 9), np.uint8) # The kernels were chosen by "guess and trial"
    eroded = cv2.erode(thresh, kernel_erosion)
    kernel_dilatation = np.ones((9, 9), np.uint8)
    return cv2.dilate(eroded, kernel_dilatation)


def main():
    # Loading and showing the 'raw' video:
    video = load_video_from_argv()
    video_frames = show_video(video)

    # Using the MOG2 algorithm to obtain the filtered image:
    background, mogged_frames = perform_bg_subtraction(video_frames)

    # Showing each 'mogged' frame:
    play_video(
        mogged_frames,
        framerate = 120, # really fast (but no so fast)
        res_func = lambda frame : frame
    )

    # Showing each treated frame:
    play_video(
        mogged_frames,
        framerate = 120, # really fast (but no so fast)
        res_func = lambda frame : treat_image(frame)
    )
    
    # The framerate is low so we can see each frame slowly
    framerate = 30
    for mogged_frame, frame in zip(mogged_frames, video_frames):
        # We treat each mogged frame:
        result = treat_image(mogged_frame)

        # I converted it to grayscale, so the findContours function will work properly
        filtered_gry = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        outlines, _ = cv2.findContours(filtered_gry, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in outlines:
            if cv2.contourArea(c) <= 2500 or cv2.contourArea(c) >= 0.8 * frame.shape[0] * frame.shape[1]:
                # I defined the minimum contour area as 5000, so no noise can be influence the bounding boxes 
                continue

            # Drawing the bounding boxes:
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow('My video', frame) # Showing each frame of the resultating video

        if cv2.waitKey(1000 // framerate) == ord('k'): # If you press 'k' the window will close
            break

    
    
if __name__ == "__main__":
    main()