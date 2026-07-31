import librosa
import numpy as np
from scipy.signal import find_peaks
from moviepy import VideoFileClip, concatenate_videoclips
import os

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

    #print(f"--> The absolute LOUDEST moment in this video is: {np.max(rms)}")
    # Create an array of timestamps corresponding to each RMS frame
    times = librosa.times_like(rms, sr=sr)

    dynamic_threshold = np.percentile(rms, 99.5)
    
    print(f"--> Average Volume (Mean): {np.mean(rms):.4f}")
    print(f"--> Loudest Moment (Max): {np.max(rms):.4f}")
    print(f"--> Auto-Calculated Threshold (Top 2%): {dynamic_threshold:.4f}")
    # ---------------------------------------------------------
    print("3. Detecting audio peaks...")
    # Find peaks where the RMS exceeds our threshold. 
    # 'distance' prevents capturing the same continuous shout as multiple distinct events.
    # We calculate the distance in frames (e.g., ensuring at least 10 seconds between peaks).
    frames_per_sec = len(rms) / video.duration
    min_distance = int(10 * frames_per_sec) 
    
    peaks, _ = find_peaks(rms, height=dynamic_threshold, distance=min_distance) 
    
    print(f"Found {len(peaks)} potential highlights. Cutting clips...")
    highlights = []
    for peak_idx in peaks:
        peak_time = times[peak_idx]
        
        # Define start and end times, ensuring we don't go out of bounds of the video duration
        start_time = max(0, peak_time - clip_before)
        end_time = min(video.duration, peak_time + clip_after)
        
        # Extract the subclip
        # Note: MoviePy v2.x uses subclipped(), v1.x uses subclip()
        try:
            clip = video.subclipped(start_time, end_time)
        except AttributeError:
            clip = video.subclip(start_time, end_time)
            
        highlights.append(clip)
        
    print("4. Stitching highlights together...")
    if highlights:
        # Concatenate all the cut clips into one final highlight reel
        final_video = concatenate_videoclips(highlights)
        final_video.write_videofile(output_path, codec="libx264", audio_codec="aac")
        print(f"Success! Highlights saved to {output_path}")
    else:
        print("No highlights found. Try lowering the threshold parameter.")

    # Clean up temporary audio file
    if os.path.exists(temp_audio_path):
        os.remove(temp_audio_path)

# Execute the function
extract_highlights("test_match.mp4", "final_highlights.mp4")