import picamera
import time

TOTAL_IMAGES = 1500 #Adjust if desired

def capture_generator():
    for i in range(TOTAL_IMAGES):
        yield f'image_{i:04d}.jpg'

with picamera.PiCamera() as camera:
    #Lower resolution is faster, 720p is good for high FPS
    camera.resolution = (1280, 720)
    
    #Camera warm-up
    camera.start_preview()
    time.sleep(2)
    
    print(f"Starting capture of {TOTAL_IMAGES} images...")
    start_time = time.time()
  
    camera.capture_sequence(capture_generator(), use_video_port=True, format='jpeg')
    
    end_time = time.time()
    
    print(f"Captured {TOTAL_IMAGES} images in {end_time - start_time:.2f} seconds")
