import picamera
import time
import cv2
import numpy as np
import os

TOTAL_IMAGES = 1500 #Change if needed
OUTPUT_DIR = "vein_masks"
os.makedirs(OUTPUT_DIR, exist_ok=True)

lower_vein = np.array([0, 0, 0], np.uint8)
upper_vein = np.array([180, 255, 50], np.uint8)

def capture_and_process():
    with picamera.PiCamera() as camera:
        camera.resolution = (1280, 720)
        camera.start_preview()
        time.sleep(2) # Warm-up

        print(f"Starting capture and masking of {TOTAL_IMAGES} images...")
        start_time = time.time()

        for i in range(TOTAL_IMAGES):
            #Capture into memory
            stream = io.BytesIO()
            camera.capture(stream, format='jpeg', use_video_port=True)
            
            data = np.frombuffer(stream.getvalue(), dtype=np.uint8)
            img = cv2.imdecode(data, 1)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            
            #Create binary mask (White = Veins, Black = Background)
            mask = cv2.inRange(hsv, lower_vein, upper_vein)

            #Clean up mask (Dilation/Erosion)
            kernel = np.ones((5,5), np.uint8)
            mask = cv2.erode(mask, kernel, iterations=1)
            mask = cv2.dilate(mask, kernel, iterations=1)

            #Save the mask
            cv2.imwrite(f"{OUTPUT_DIR}/mask_{i:04d}.png", mask)
            
            if i % 100 == 0: print(f"Processed {i} images...")

        end_time = time.time()
        print(f"Finished in {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    import io
    capture_and_process()
