
![project_banner](https://github.com/user-attachments/assets/0db918da-732f-40d4-81fa-14c1cddc30b6)

# PoseVision: Real-Time Landmark Tracker

## Overview
PoseVision is a real-time computer vision tool that tracks and displays human landmarks—like face contours, hands, and full-body poses—using your webcam. Built with Python and MediaPipe, it comes with a sleek, modern GUI powered by CustomTkinter. You can capture images, record videos in MP4 format, and even see basic actions (like raising a hand) detected live on-screen. The code is clean, handles errors well, and has plenty of notes to help you dig in.

## Features
- **Live Landmark Detection**: Tracks face, hands, and body poses in real-time.
- **Modern GUI**: Dark-themed interface with a smooth, user-friendly design.
- **Image Capture**: Snap photos (JPG) when a pose is detected.
- **Video Recording**: Save MP4 videos with landmarks included.
- **Action Detection**: Displays simple actions like "Standing" or "Raising Hand".
- **Error Handling**: Keeps things running smoothly with logging.
- **Customizable**: Easy to tweak colors, settings, or add new features.

## Requirements
- Python 3.7+
- Dependencies (see `requirements.txt`):
  - `mediapipe>=0.8.9`
  - `opencv-python>=4.5.5`
  - `numpy>=1.21.0`
  - `Pillow>=9.0.0`
  - `customtkinter>=5.2.0`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/[your-username]/posevision.git
   cd posevision
   ```
2. Set up a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

# Usage
1. Run the app:
   ```bash
   python posevision.py
   ```
2. What you’ll see:
  - A live video feed with landmarks.
  - Buttons for "Capture Image" and "Start/Stop Recording".
  - A status bar showing detection and recording updates.
  - An "Action" label showing what you’re doing (e.g., "Raising Right Hand").
    
3. How to use it:
 - Click "Capture Image" to save a JPG when a pose is detected.
 - Click "Start Recording" to save an MP4; click again to stop.
 - Close the window to exit.

# Project Structure
 - posevision.py: Main script with all the code.
 - requirements.txt: List of dependencies.
 - README.md: This file.

# How It Works
PoseVision grabs your webcam feed with OpenCV, processes it with MediaPipe to find landmarks, and shows everything in a fancy GUI. It can:
 - Draw landmarks on your face, hands, and body.
 - Save images or videos with timestamps.
 - Guess simple actions by checking where your hands are compared to your shoulders.

<img src="https://github.com/user-attachments/assets/7f4a45e2-a758-4e71-a1d3-f6dad45cf780" width="600" height="450">

# Notes
 - Make sure your webcam is plugged in and working.
 - Good lighting helps with detection.
 - Videos and images save in the same folder as the script—edit the code to change that if you want.
 - The action detection is basic (e.g., hand raises); you can make it smarter with more rules!

# License
This project is licensed under the MIT License.
```bash
MIT License

Copyright (c) 2025 Mohammad Hosein Akrami

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```
