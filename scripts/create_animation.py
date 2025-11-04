"""
Create an animated GIF from the saved visualization frames.
"""

import os
from PIL import Image


def create_gif(frames_dir="visualizations", output_file="truck_animation.gif", duration=500):
    """
    Create an animated GIF from PNG frames.
    
    Args:
        frames_dir: Directory containing the frame PNGs
        output_file: Output GIF filename
        duration: Duration of each frame in milliseconds
    """
    # Get all PNG files sorted by name
    frame_files = sorted([
        f for f in os.listdir(frames_dir) 
        if f.endswith('.png') and f.startswith('step_')
    ])
    
    if not frame_files:
        print(f"❌ No frame files found in {frames_dir}/")
        return
    
    print(f"🎬 Found {len(frame_files)} frames")
    print(f"📁 Reading frames from: {frames_dir}/")
    
    # Load images
    frames = []
    for frame_file in frame_files:
        frame_path = os.path.join(frames_dir, frame_file)
        img = Image.open(frame_path)
        frames.append(img)
        print(f"  ✓ Loaded {frame_file}")
    
    # Save as GIF
    print(f"\n💾 Creating animation: {output_file}")
    frames[0].save(
        output_file,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0  # Loop forever
    )
    
    # Get file size
    file_size = os.path.getsize(output_file)
    file_size_mb = file_size / (1024 * 1024)
    
    print(f"✅ Animation created successfully!")
    print(f"   📄 File: {output_file}")
    print(f"   📊 Size: {file_size_mb:.2f} MB")
    print(f"   🎞️  Frames: {len(frames)}")
    print(f"   ⏱️  Duration per frame: {duration}ms")
    print(f"   🔄 Total duration: {len(frames) * duration / 1000:.1f}s")


if __name__ == "__main__":
    print("\n" + "="*80)
    print("🎥 TRUCK ROUTING ANIMATION CREATOR")
    print("="*80 + "\n")
    
    create_gif(
        frames_dir="visualizations",
        output_file="truck_animation.gif",
        duration=500  # 500ms = 0.5s per frame
    )
    
    print("\n" + "="*80)
    print("You can now view the animation by opening truck_animation.gif")
    print("="*80 + "\n")
