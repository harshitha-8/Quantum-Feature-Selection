from PIL import Image, ImageDraw, ImageFont
import os
from pathlib import Path

def load_tahoma(size):
    font_paths = [
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/Library/Fonts/Tahoma.ttf",
        "/usr/share/fonts/truetype/tahoma.ttf",
        "/System/Library/Fonts/Tahoma.ttf"
    ]
    for p in font_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def wrap_text(text, font, max_width, draw):
    words = text.split()
    lines = []
    current = words[0]
    for word in words[1:]:
        if draw.textbbox((0,0), f"{current} {word}", font=font)[2] <= max_width:
            current = f"{current} {word}"
        else:
            lines.append(current)
            current = word
    lines.append(current)
    return "\n".join(lines)

def main():
    root = Path("/Volumes/T9/QuantumFeatureSelection")
    demo_dir = root / "cvpr demo"
    
    # Create 1920x1080 canvas
    canvas = Image.new("RGB", (1920, 1080), "white")
    draw = ImageDraw.Draw(canvas)
    
    # Use 25 * 3 = 75pt font so it mimics a "25pt" font size on a 1080p screen
    tf = load_tahoma(75)
    
    text = "QAgroVis: Real-Time Cotton Defoliation Monitoring via Quantum-Classical Hybrid Feature Selection on Temporal UAV Imagery"
    wrapped = wrap_text(text, tf, 1700, draw)
    
    # Calculate textbox dynamically to securely center it
    bbox = draw.multiline_textbbox((0, 0), wrapped, font=tf, align="center")
    x = (1920 - (bbox[2] - bbox[0])) // 2
    y = (1080 - (bbox[3] - bbox[1])) // 2
    
    draw.multiline_text((x, y), wrapped, font=tf, fill="#111111", align="center")
    
    out_path = demo_dir / "00_title_slide.png"
    canvas.save(out_path, format="PNG")
    print(f"Saved {out_path}")

if __name__ == '__main__':
    main()
