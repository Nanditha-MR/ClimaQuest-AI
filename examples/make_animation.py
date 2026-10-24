"""
ClimaQuest AI — Member 2 Utility: Mission Control Animation Generator
Compiles rendered surveillance frames into an animated GIF for presentations and UI.
"""

import os
import glob
from PIL import Image

def generate_animation(fps: int = 2):
    frames_dir = os.path.join(os.path.dirname(__file__), "..", "output", "frames")
    frame_files = sorted(glob.glob(os.path.join(frames_dir, "frame_*.png")))
    
    if not frame_files:
        print(f"No frames found in {frames_dir}. Run examples/run_demo.py first.")
        return

    images = [Image.open(f) for f in frame_files]
    output_gif = os.path.join(os.path.dirname(__file__), "..", "output", "mission_control_surveillance.gif")
    
    duration_ms = int(1000 / fps)
    images[0].save(
        output_gif,
        save_all=True,
        append_images=images[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
    )
    print(f"Animated mission control GIF saved to: {output_gif}")
    print(f"Total frames: {len(images)} | Frame duration: {duration_ms} ms")

if __name__ == "__main__":
    generate_animation(fps=2)
