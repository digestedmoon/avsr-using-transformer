import os
import cv2
import subprocess

def auto_preprocess_video(input_mp4: str, output_mp4: str) -> bool:
    """
    Takes a raw video file, detects a face in the first frame,
    calculates a bounding box around the mouth, and uses ffmpeg
    to crop, scale to 160x160, convert to 25 FPS, and extract 16kHz mono audio.
    
    Returns True if successful, False if no face was detected or an error occurred.
    """
    
    cap = cv2.VideoCapture(input_mp4)
    if not cap.isOpened():
        print(f"[AutoCrop] Error: Cannot open video {input_mp4}")
        return False
        
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print(f"[AutoCrop] Error: Cannot read first frame of {input_mp4}")
        return False

    orig_h, orig_w = frame.shape[:2]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(50, 50)
    )
    
    if len(faces) == 0:
        print("[AutoCrop] Warning: No face detected. Defaulting to center crop.")
        crop_size = min(orig_w, orig_h)
        cx, cy = orig_w // 2, orig_h // 2
    else:
        faces = sorted(faces, key=lambda f: f[2]*f[3], reverse=True)
        x, y, w, h = faces[0]
        
        cx = x + w // 2
        cy = y + int(0.7 * h)
        
        crop_size = int(w * 1.5)

    crop_x = cx - crop_size // 2
    crop_y = cy - crop_size // 2
    
    if crop_x < 0:
        crop_x = 0
    if crop_y < 0:
        crop_y = 0
    if crop_x + crop_size > orig_w:
        crop_size = orig_w - crop_x
    if crop_y + crop_size > orig_h:
        crop_size = min(crop_size, orig_h - crop_y)
        
    crop_size = min(crop_size, orig_w - crop_x, orig_h - crop_y)

    print(f"[AutoCrop] Crop calculated: x={crop_x}, y={crop_y}, size={crop_size}x{crop_size}")
    
    ffmpeg_cmd = [
        "ffmpeg", "-y", "-v", "quiet",
        "-i", input_mp4,
        "-vf", f"crop={crop_size}:{crop_size}:{crop_x}:{crop_y},scale=160:160,fps=25",
        "-ac", "1",
        "-ar", "16000",
        "-vcodec", "libx264",
        "-strict", "experimental",
        output_mp4
    ]
    
    try:
        subprocess.run(ffmpeg_cmd, check=True)
        print(f"[AutoCrop] Successfully formatted video to {output_mp4}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"[AutoCrop] FFmpeg formatting failed: {e}")
        return False

