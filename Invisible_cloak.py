import cv2
import numpy as np
import time

# Function to capture the background
def capture_background(cap, num_frames=10):
    time.sleep(3)  # Giving some time for the camera to warm-up
    for i in range(num_frames):
        ret, background = cap.read()
    return background

# Function to process the frame and create the invisible cloak effect
def process_frame(frame, background):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define range for blue color detection
    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])
    
    # Create a mask to detect blue color
    mask1 = cv2.inRange(hsv, lower_blue, upper_blue)
    
    # Refine the mask (Optional, to make it more smooth)
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    mask1 = cv2.morphologyEx(mask1, cv2.MORPH_DILATE, np.ones((3,3), np.uint8))
    
    # Create an inverse mask
    mask2 = cv2.bitwise_not(mask1)
    
    # Segment the blue color part out of the frame using bitwise and with the inverted mask
    res1 = cv2.bitwise_and(frame, frame, mask=mask2)
    
    # Create the background part only for the blue color area
    res2 = cv2.bitwise_and(background, background, mask=mask1)
    
    # Generate the final output by combining res1 and res2
    final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
    
    return final_output

def main():
    cap = cv2.VideoCapture(0)
    
    print("Capturing background...")
    background = capture_background(cap)
    
    print("Processing frames...")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        final_output = process_frame(frame, background)
        
        cv2.imshow('Invisible Cloak', final_output)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()