import librosa
import numpy as np
from scipy.signal import find_peaks
from moviepy import VideoFileClip, concatenate_videoclips
from scenedetect import detect, ContentDetector
import os

def find_last_scene_cut(video_path, end_time, search_window=10):
    """
    Looks backward from the peak volume to find the last camera cut.
    """
    # Start looking a few seconds before the peak
    start_time = max(0, end_time - search_window)
    
    # Detect scenes using PySceneDetect
    # ContentDetector looks for fast visual changes (like a broadcast camera cut)
    scene_list = detect(video_path, ContentDetector(threshold=27.0), start_time=start_time, end_time=end_time)
    
    # If it finds camera cuts, return the start time of the very last cut
    if scene_list:
        # scene_list[-1] is the last scene before the loud noise
        # [0] gets the start time of that scene, .get_seconds() converts to float
        return scene_list[-1][0].seconds
    
    # If no cut is found, default to a standard 5 seconds before
    return max(0, end_time - 5)
def extract_highlights(video_path, output_path, threshold=0.15, clip_before=5, clip_after=5):
    print("1. Loading video and extracting audio...")
    # Load the video file
    video = VideoFileClip(video_path)
    
    # Export audio to a temporary WAV file for Librosa to process efficiently
    temp_audio_path = "temp_audio.wav"
    video.audio.write_audiofile(temp_audio_path, logger=None)
    
    print("2. Computing RMS loudness...")
    # Load the audio using Librosa (sr=None preserves the original sample rate)
    y, sr = librosa.load(temp_audio_path, sr=None)
    
    # Calculate Root-Mean-Square (RMS) energy for each frame
    rms = librosa.feature.rms(y=y)[0]

    # Create an array of timestamps corresponding to each RMS frame
    times = librosa.times_like(rms, sr=sr)

    dynamic_threshold = np.percentile(rms, 99.5)
    
    print(f"--> Average Volume (Mean): {np.mean(rms):.4f}")
    print(f"--> Loudest Moment (Max): {np.max(rms):.4f}")
    print(f"--> Auto-Calculated Threshold (Top 0.5%): {dynamic_threshold:.4f}")
    # ---------------------------------------------------------
    print("3. Detecting audio peaks...")
    frames_per_sec = len(rms) / video.duration
    min_distance = int(10 * frames_per_sec) 
    
    peaks, _ = find_peaks(rms, height=dynamic_threshold, distance=min_distance) 
    
    print(f"Found {len(peaks)} potential highlights. Detecting visual scene changes for clean cuts...")
    highlights = []
    
    for i, peak_idx in enumerate(peaks):
        peak_time = times[peak_idx]
        print(f"   Analyzing visual cut for highlight {i+1} at {peak_time:.1f}s...")
        
        # 1. Vision: Find the exact camera cut before the loud noise
        start_time = find_last_scene_cut(video_path, peak_time)
        
        # 2. Add time after the peak to catch the celebration
        end_time = min(video.duration, peak_time + clip_after)
        
        # 3. Safeguard: If the camera cut happened less than 2 seconds before the goal, 
        # force the standard clip_before fallback so we don't miss the build-up.
        if (peak_time - start_time) < 2:
            start_time = max(0, peak_time - clip_before)
            
        # Extract the subclip
        try:
            clip = video.subclipped(start_time, end_time)
        except AttributeError:
            clip = video.subclip(start_time, end_time)
            
        highlights.append(clip)
        
    print("4. Stitching highlights together...")
    if highlights:
        final_video = concatenate_videoclips(highlights)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Success! Highlights saved to {output_path}")
    else:
        print("No highlights found. Try lowering the threshold parameter.")

    # Clean up temporary audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)
# Execute the function
extract_highlights("test_match2.mp4", "final_highlights.mp4")