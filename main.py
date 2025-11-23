"""
Dental Camera Display Software
USB Camera feed with controls for patient hall TV display
"""

import cv2
import numpy as np
from tkinter import Tk, Label, Button, Scale, Frame, HORIZONTAL, messagebox
from PIL import Image, ImageTk
import sys

class DentalCameraApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Dental Camera - Control Panel")
        self.root.geometry("1200x800")
        
        # Camera variables
        self.cap = None
        self.camera_index = 0
        self.is_running = False
        self.is_frozen = False
        self.frozen_frame = None
        
        # Image adjustments
        self.brightness = 0
        self.contrast = 1.0
        self.zoom_level = 1.0
        self.flip_horizontal = False
        self.flip_vertical = False
        
        # Setup UI
        self.setup_ui()
        
        # Try to initialize camera
        self.initialize_camera()
        
        # Start video loop
        if self.is_running:
            self.update_frame()
    
    def setup_ui(self):
        # Main container
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Video display area
        self.video_label = Label(main_frame, bg="black")
        self.video_label.pack(side="left", fill="both", expand=True)
        
        # Control panel
        control_frame = Frame(main_frame, width=300)
        control_frame.pack(side="right", fill="y", padx=(10, 0))
        
        # Title
        Label(control_frame, text="Camera Controls", font=("Arial", 16, "bold")).pack(pady=10)
        
        # Freeze/Unfreeze button
        self.freeze_btn = Button(control_frame, text="Freeze Frame", 
                                command=self.toggle_freeze, 
                                bg="#4CAF50", fg="white", 
                                font=("Arial", 12), height=2)
        self.freeze_btn.pack(fill="x", pady=5)
        
        # Brightness control
        Label(control_frame, text="Brightness", font=("Arial", 10)).pack(pady=(15, 5))
        self.brightness_scale = Scale(control_frame, from_=-100, to=100, 
                                     orient=HORIZONTAL, command=self.update_brightness)
        self.brightness_scale.set(0)
        self.brightness_scale.pack(fill="x")
        
        # Contrast control
        Label(control_frame, text="Contrast", font=("Arial", 10)).pack(pady=(15, 5))
        self.contrast_scale = Scale(control_frame, from_=0.5, to=3.0, 
                                   resolution=0.1, orient=HORIZONTAL, 
                                   command=self.update_contrast)
        self.contrast_scale.set(1.0)
        self.contrast_scale.pack(fill="x")
        
        # Zoom control
        Label(control_frame, text="Digital Zoom", font=("Arial", 10)).pack(pady=(15, 5))
        self.zoom_scale = Scale(control_frame, from_=1.0, to=3.0, 
                               resolution=0.1, orient=HORIZONTAL, 
                               command=self.update_zoom)
        self.zoom_scale.set(1.0)
        self.zoom_scale.pack(fill="x")
        
        # Flip controls
        Button(control_frame, text="Flip Horizontal", 
               command=self.toggle_flip_h, height=2).pack(fill="x", pady=5)
        Button(control_frame, text="Flip Vertical", 
               command=self.toggle_flip_v, height=2).pack(fill="x", pady=5)
        
        # Reset button
        Button(control_frame, text="Reset All Settings", 
               command=self.reset_settings, 
               bg="#FF9800", fg="white", 
               font=("Arial", 10), height=2).pack(fill="x", pady=(15, 5))
        
        # Fullscreen button
        Button(control_frame, text="Open Fullscreen Display", 
               command=self.open_fullscreen, 
               bg="#2196F3", fg="white", 
               font=("Arial", 12), height=2).pack(fill="x", pady=(15, 5))
        
        # Exit button
        Button(control_frame, text="Exit", 
               command=self.on_closing, 
               bg="#f44336", fg="white", 
               font=("Arial", 12), height=2).pack(fill="x", pady=5)
        
        # Status label
        self.status_label = Label(control_frame, text="Status: Initializing...", 
                                 font=("Arial", 9), fg="gray")
        self.status_label.pack(side="bottom", pady=10)
        
        # Handle window closing
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
    
    def initialize_camera(self):
        """Initialize USB camera"""
        for i in range(5):  # Try first 5 camera indices
            self.cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)  # DSHOW for Windows
            if self.cap.isOpened():
                self.camera_index = i
                self.is_running = True
                
                # Set camera properties for better quality
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                self.cap.set(cv2.CAP_PROP_FPS, 30)
                
                self.status_label.config(text=f"Status: Camera {i} connected", fg="green")
                return True
        
        self.status_label.config(text="Status: No camera found!", fg="red")
        messagebox.showerror("Camera Error", 
                           "No USB camera detected!\nPlease connect camera and restart.")
        return False
    
    def update_brightness(self, value):
        self.brightness = int(float(value))
    
    def update_contrast(self, value):
        self.contrast = float(value)
    
    def update_zoom(self, value):
        self.zoom_level = float(value)
    
    def toggle_flip_h(self):
        self.flip_horizontal = not self.flip_horizontal
    
    def toggle_flip_v(self):
        self.flip_vertical = not self.flip_vertical
    
    def toggle_freeze(self):
        if not self.is_frozen:
            self.is_frozen = True
            self.freeze_btn.config(text="Unfreeze", bg="#FF5722")
        else:
            self.is_frozen = False
            self.frozen_frame = None
            self.freeze_btn.config(text="Freeze Frame", bg="#4CAF50")
    
    def reset_settings(self):
        self.brightness_scale.set(0)
        self.contrast_scale.set(1.0)
        self.zoom_scale.set(1.0)
        self.flip_horizontal = False
        self.flip_vertical = False
        self.is_frozen = False
        self.frozen_frame = None
        self.freeze_btn.config(text="Freeze Frame", bg="#4CAF50")
    
    def process_frame(self, frame):
        """Apply all adjustments to frame"""
        # Apply brightness and contrast
        frame = cv2.convertScaleAbs(frame, alpha=self.contrast, beta=self.brightness)
        
        # Apply zoom
        if self.zoom_level > 1.0:
            h, w = frame.shape[:2]
            crop_h, crop_w = int(h / self.zoom_level), int(w / self.zoom_level)
            start_h, start_w = (h - crop_h) // 2, (w - crop_w) // 2
            frame = frame[start_h:start_h + crop_h, start_w:start_w + crop_w]
            frame = cv2.resize(frame, (w, h))
        
        # Apply flips
        if self.flip_horizontal:
            frame = cv2.flip(frame, 1)
        if self.flip_vertical:
            frame = cv2.flip(frame, 0)
        
        return frame
    
    def update_frame(self):
        """Main video loop"""
        if not self.is_running:
            return
        
        if self.is_frozen and self.frozen_frame is not None:
            frame = self.frozen_frame
        else:
            ret, frame = self.cap.read()
            if not ret:
                self.status_label.config(text="Status: Camera disconnected!", fg="red")
                self.root.after(1000, self.initialize_camera)  # Try to reconnect
                return
            
            frame = self.process_frame(frame)
            
            if self.is_frozen and self.frozen_frame is None:
                self.frozen_frame = frame.copy()
        
        # Convert to PhotoImage for display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit display
        display_width = 900
        display_height = int(display_width * img.height / img.width)
        img = img.resize((display_width, display_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo
        
        # Schedule next frame
        self.root.after(33, self.update_frame)  # ~30 FPS
    
    def open_fullscreen(self):
        """Open fullscreen window on secondary display"""
        fullscreen_window = FullscreenDisplay(self)
    
    def on_closing(self):
        """Cleanup and exit"""
        self.is_running = False
        if self.cap is not None:
            self.cap.release()
        self.root.quit()
        self.root.destroy()


class FullscreenDisplay:
    """Fullscreen window for patient hall TV"""
    def __init__(self, parent_app):
        self.parent = parent_app
        self.window = Tk()
        self.window.title("Dental Camera - Patient Display")
        self.window.attributes('-fullscreen', True)
        self.window.configure(bg='black')
        
        # Video display
        self.video_label = Label(self.window, bg="black")
        self.video_label.pack(fill="both", expand=True)
        
        # Exit fullscreen with Escape
        self.window.bind('<Escape>', lambda e: self.close_window())
        
        # Start update loop
        self.update_frame()
        self.window.mainloop()
    
    def update_frame(self):
        """Update fullscreen display"""
        if not self.parent.is_running:
            return
        
        if self.parent.is_frozen and self.parent.frozen_frame is not None:
            frame = self.parent.frozen_frame
        else:
            ret, frame = self.parent.cap.read()
            if not ret:
                self.window.after(100, self.update_frame)
                return
            
            frame = self.parent.process_frame(frame)
        
        # Convert and display
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame_rgb)
        
        # Resize to fit screen
        screen_width = self.window.winfo_screenwidth()
        screen_height = self.window.winfo_screenheight()
        img = img.resize((screen_width, screen_height), Image.Resampling.LANCZOS)
        
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo)
        self.video_label.image = photo
        
        self.window.after(33, self.update_frame)
    
    def close_window(self):
        self.window.destroy()


def main():
    root = Tk()
    app = DentalCameraApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
