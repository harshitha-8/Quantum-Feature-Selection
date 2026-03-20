import cv2
import numpy as np
from pathlib import Path

def resize_and_pad(img, target_size=(1920, 1080), pad_color=(255, 255, 255)):
    h, w = img.shape[:2]
    tw, th = target_size
    scale = min(tw/w, th/h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    top = (th - new_h) // 2
    bottom = th - new_h - top
    left = (tw - new_w) // 2
    right = tw - new_w - left
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=pad_color)
    return padded

def main():
    root_dir = Path("/Volumes/T9/QuantumFeatureSelection")
    demo_dir = root_dir / "cvpr demo"
    video_dir = demo_dir / "video"
    video_dir.mkdir(parents=True, exist_ok=True)
    
    # Grab all properly ordered 01_ through 18_ files
    images = sorted(list(demo_dir.glob("*.png")))
    if not images:
        print("No images found in cvpr demo!")
        return
        
    out_path = video_dir / "cvpr_demo_presentation.mp4"
    
    # Standard mp4v codec for cross-system MacOS playback
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    fps = 30
    frames_per_image = 60  # 2 seconds per image slide
    target_size = (1920, 1080)
    
    out = cv2.VideoWriter(str(out_path), fourcc, fps, target_size)
    
    print(f"Generating video with {len(images)} slides...")
    for img_path in images:
        print(f"Transcoding {img_path.name}...")
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        padded = resize_and_pad(img, target_size)
        for _ in range(frames_per_image):
            out.write(padded)
            
    out.release()
    print(f"Video saved successfully! Total size: {out_path.stat().st_size / (1024*1024):.2f} MB")

if __name__ == '__main__':
    main()
