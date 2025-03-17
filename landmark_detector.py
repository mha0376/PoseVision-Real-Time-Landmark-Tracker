"""
Landmark Detection System with Modern GUI and Action Detection
Author: Mohammad Hosein Akrami
Date: March 16, 2025
Description: A real-time human landmark detection system with a modern GUI, including image capture, MP4 video recording, and action name display.
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple
import logging
import customtkinter as ctk
from PIL import Image, ImageTk
import datetime
import os


class LandmarkDetectorGUI:
    """A modern GUI-based class for detecting and visualizing human landmarks with image, video, and action features."""

    def __init__(self,
                 min_detection_confidence: float = 0.5,
                 min_tracking_confidence: float = 0.5,
                 camera_index: int = 0):
        """Initialize the LandmarkDetectorGUI with detection settings and modern GUI components."""
        # MediaPipe components
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )

        # Video capture
        self.cap = cv2.VideoCapture(camera_index)
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # Default to 30 if FPS unavailable
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self._setup_logging()
        self._init_drawing_specs()

        # Video recording variables
        self.recording = False
        self.video_writer = None

        # Modern GUI setup with CustomTkinter
        ctk.set_appearance_mode("dark")  # Dark theme
        ctk.set_default_color_theme("blue")  # Blue accents

        self.root = ctk.CTk()
        self.root.title("Landmark Detection System")
        self.root.geometry("900x700")
        self.root.resizable(False, False)

        # Frame for video display
        self.video_frame = ctk.CTkFrame(self.root, width=640, height=480)
        self.video_frame.pack(pady=20)

        # Video display label
        self.video_label = ctk.CTkLabel(self.video_frame, text="")
        self.video_label.pack()

        # Control frame for buttons and status
        self.control_frame = ctk.CTkFrame(self.root)
        self.control_frame.pack(pady=10)

        # Capture image button
        self.capture_button = ctk.CTkButton(
            self.control_frame, text="Capture Image", command=self.capture_image,
            width=200, height=40, font=("Arial", 16)
        )
        self.capture_button.pack(side="left", padx=10, pady=10)

        # Record video button
        self.record_button = ctk.CTkButton(
            self.control_frame, text="Start Recording", command=self.toggle_recording,
            width=200, height=40, font=("Arial", 16)
        )
        self.record_button.pack(side="left", padx=10, pady=10)

        # Status frame for status and action
        self.status_frame = ctk.CTkFrame(self.root)
        self.status_frame.pack(pady=10)

        # Status label
        self.status_label = ctk.CTkLabel(
            self.status_frame, text="Ready", font=("Arial", 14), text_color="gray"
        )
        self.status_label.pack(side="left", padx=20)

        # Action name label
        self.action_label = ctk.CTkLabel(
            self.status_frame, text="Action: None", font=("Arial", 14, "bold"), text_color="cyan"
        )
        self.action_label.pack(side="left", padx=20)

        # Running flag
        self.running = True

    def _setup_logging(self) -> None:
        """Configure logging for the application."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def _init_drawing_specs(self) -> None:
        """Initialize drawing specifications for landmarks."""
        self.face_spec = (
            self.mp_drawing.DrawingSpec(
                color=(80, 110, 10), thickness=1, circle_radius=1),
            self.mp_drawing.DrawingSpec(
                color=(80, 256, 121), thickness=1, circle_radius=1)
        )
        self.right_hand_spec = (
            self.mp_drawing.DrawingSpec(
                color=(80, 22, 10), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(
                color=(80, 44, 121), thickness=2, circle_radius=2)
        )
        self.left_hand_spec = (
            self.mp_drawing.DrawingSpec(
                color=(121, 22, 76), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(
                color=(121, 44, 250), thickness=2, circle_radius=2)
        )
        self.pose_spec = (
            self.mp_drawing.DrawingSpec(
                color=(245, 117, 66), thickness=2, circle_radius=4),
            self.mp_drawing.DrawingSpec(
                color=(245, 66, 230), thickness=2, circle_radius=2)
        )

    def process_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, Optional[object]]:
        """Process a frame for landmark detection."""
        try:
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.holistic.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            self._draw_landmarks(image, results)
            return image, results
        except Exception as e:
            self.logger.error(f"Error processing frame: {str(e)}")
            self.status_label.configure(
                text=f"Error: {str(e)}", text_color="red")
            return frame, None

    def _draw_landmarks(self, image: np.ndarray, results: object) -> None:
        """Draw detected landmarks on the image."""
        if results.face_landmarks:
            self.mp_drawing.draw_landmarks(image, results.face_landmarks,
                                           self.mp_holistic.FACEMESH_CONTOURS, *self.face_spec)
        if results.right_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                                           self.mp_holistic.HAND_CONNECTIONS, *self.right_hand_spec)
        if results.left_hand_landmarks:
            self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                                           self.mp_holistic.HAND_CONNECTIONS, *self.left_hand_spec)
        if results.pose_landmarks:
            self.mp_drawing.draw_landmarks(image, results.pose_landmarks,
                                           self.mp_holistic.POSE_CONNECTIONS, *self.pose_spec)

    def detect_action(self, results: object) -> str:
        """Detect basic actions based on pose landmarks."""
        if not results or not results.pose_landmarks:
            return "None"

        landmarks = results.pose_landmarks.landmark
        # Example action detection based on landmark positions
        right_shoulder = landmarks[self.mp_holistic.PoseLandmark.RIGHT_SHOULDER.value].y
        right_wrist = landmarks[self.mp_holistic.PoseLandmark.RIGHT_WRIST.value].y
        left_shoulder = landmarks[self.mp_holistic.PoseLandmark.LEFT_SHOULDER.value].y
        left_wrist = landmarks[self.mp_holistic.PoseLandmark.LEFT_WRIST.value].y

        if right_wrist < right_shoulder - 0.2:  # Right hand raised above shoulder
            return "Raising Right Hand"
        elif left_wrist < left_shoulder - 0.2:  # Left hand raised above shoulder
            return "Raising Left Hand"
        else:
            return "Standing"

    def capture_image(self) -> None:
        """Capture and save the current frame if pose landmarks are detected."""
        if hasattr(self, 'last_results') and self.last_results.pose_landmarks:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"capture_{timestamp}.jpg"
            cv2.imwrite(filename, self.last_frame)
            self.logger.info(f"Image saved as {filename}")
            self.status_label.configure(
                text=f"Saved: {filename}", text_color="green")
        else:
            self.logger.warning("No pose landmarks detected to capture!")
            self.status_label.configure(
                text="No pose detected!", text_color="orange")

    def toggle_recording(self) -> None:
        """Start or stop video recording."""
        if not self.recording:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"recording_{timestamp}.mp4"
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 codec
            self.video_writer = cv2.VideoWriter(filename, fourcc, self.fps,
                                                (self.frame_width, self.frame_height))
            self.recording = True
            self.record_button.configure(text="Stop Recording")
            self.status_label.configure(
                text=f"Recording: {filename}", text_color="red")
            self.logger.info(f"Started recording: {filename}")
        else:
            self.recording = False
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None
            self.record_button.configure(text="Start Recording")
            self.status_label.configure(
                text="Recording stopped", text_color="green")
            self.logger.info("Stopped recording")

    def update_frame(self) -> None:
        """Update the GUI with the latest video frame, handle recording, and display action."""
        if self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                processed_frame, results = self.process_frame(frame)
                self.last_frame = processed_frame.copy()  # Store for capture
                self.last_results = results  # Store detection results

                # Write frame to video if recording
                if self.recording and self.video_writer:
                    self.video_writer.write(processed_frame)

                # Detect and display action
                action = self.detect_action(results)
                self.action_label.configure(text=f"Action: {action}")

                # Convert to RGB and resize for GUI
                img = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(img)
                img = img.resize((640, 480), Image.Resampling.LANCZOS)
                imgtk = ImageTk.PhotoImage(image=img)

                self.video_label.configure(image=imgtk)
                self.video_label.image = imgtk  # Keep reference to avoid garbage collection

                # Update status if not recording
                if not self.recording:
                    if results and results.pose_landmarks:
                        self.status_label.configure(
                            text="Pose detected", text_color="green")
                    else:
                        self.status_label.configure(
                            text="No pose detected", text_color="gray")

            self.root.after(10, self.update_frame)  # Schedule next update

    def run(self) -> None:
        """Run the modern GUI application."""
        self.logger.info(
            "Starting landmark detection GUI. Close window to quit.")
        self.update_frame()
        # Handle window close
        self.root.protocol("WM_DELETE_WINDOW", self.cleanup)
        self.root.mainloop()

    def cleanup(self) -> None:
        """Release resources and cleanup."""
        self.logger.info("Shutting down landmark detection system")
        self.running = False
        if self.recording and self.video_writer:
            self.video_writer.release()
        self.cap.release()
        self.holistic.close()
        self.root.destroy()


def main():
    """Main entry point for the GUI application."""
    try:
        app = LandmarkDetectorGUI()
        app.run()
    except Exception as e:
        logging.error(f"Application failed to start: {str(e)}")


if __name__ == "__main__":
    main()
