import time
import subprocess

print("Get ready! Capturing image in:")
for i in range(5, 0, -1):
    print(f"{i}...")
    time.sleep(1)

print("Capturing frame now ")

subprocess.run([
    "libcamera-still",
    "--width", "640",
    "--height", "640",
    "-o", "hand.jpg"
], check=True)

print("Stage 0 complete: hand.jpg captured successfully")
