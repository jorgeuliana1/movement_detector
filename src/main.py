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
        if cv2.waitKey(1000 // framerate) == ord('k'):
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
    for frame in video_frames:
        img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        mog2.apply(img) # This works like an "average"

    # Retuning the obtained background:
    return cv2.cvtColor(mog2.getBackgroundImage(), cv2.COLOR_BGR2RGB)

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
    kernel_erosion = np.ones((5, 5), np.uint8) # The kernels were chosen by "guess and trial"
    eroded = cv2.erode(thresh, kernel_erosion)
    kernel_dilatation = np.ones((1, 1), np.uint8)
    return cv2.dilate(eroded, kernel_dilatation)


def main():
    # Loading and showing the 'raw' video:
    video = load_video_from_argv()
    video_frames = show_video(video)

    # Showing the obtained background:
    background = perform_bg_subtraction(video_frames)
    cv2.imshow('Obtained background', background)
    cv2.waitKey(0)

    # Showing each bg subtracted frame:
    play_video(
        video_frames,
        framerate = 120, # really fast (but no so fast)
        res_func = lambda frame : frame - background
    )

    # Performing the specified operations in the image
    dilated_bg = treat_image(background)

    # Showing each treated frame:
    play_video(
        video_frames,
        framerate = 120, # really fast (but no so fast)
        res_func = lambda frame : treat_image(frame) - dilated_bg
    )

    # These color are defined in order to be used later in the "filtering"
    clear_gray = np.array([128, 128, 128])
    white = np.array([255, 255, 255])
    
    # The framerate is low so we can see each frame slowly
    framerate = 30
    for frame in video_frames:
        # We treat each frame, as we treated the background, so they are "compatible":
        dilated_frame = treat_image(frame)

        # The subtraction will remove any intersection (the background will be removed):
        result = dilated_frame - dilated_bg

        # Observing the "result" image, I noticed that the white-ish colors are usually
        # present where humans/dogs can be seen. So I filtered it...
        mask = cv2.inRange(result, clear_gray, white) 
        filtered_res = cv2.bitwise_and(result, result, mask=mask) # Only white-ish colors are present now

        # I converted it to grayscale, so the findContours function will work properly
        filtered_gry = cv2.cvtColor(filtered_res, cv2.COLOR_BGR2GRAY).astype(np.uint8)
        blurred_res = cv2.blur(filtered_gry, (7, 7)) # I blurred it so the human presence area is "spread" (just a little)
        outlines, _ = cv2.findContours(blurred_res, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        for c in outlines:
            if cv2.contourArea(c) <= 5000:
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