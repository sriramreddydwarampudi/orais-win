import sys
import os
import warnings
import cv2
import numpy as np
import tensorflow as tf
from datetime import datetime
from pathlib import Path
from PySide6 import QtCore, QtWidgets, QtGui
from PySide6.QtGui import QImage

# 🚫 Suppress TensorFlow tflite warning
warnings.filterwarnings("ignore", category=UserWarning, module="tensorflow")

# 🔧 Get application path (works for both script and EXE)
if getattr(sys, 'frozen', False):
    # Running as compiled exe
    application_path = os.path.dirname(sys.executable)
else:
    # Running as script
    application_path = os.path.dirname(os.path.abspath(__file__))

# 🔧 Load TFLite model from application directory
model_path = os.path.join(application_path, "tooth_float32.tflite")

try:
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
except Exception as e:
    print(f"Error loading model: {e}")
    print(f"Looking for model at: {model_path}")
    sys.exit(1)

CLASS_NAMES = [
    "Normal",
    "Initial Caries",
    "Moderate Caries",
    "Severe Caries",
    "Tooth Stain",
    "Dental Calculus",
    "Other Lesions"
]

# 📁 Create save directories (Windows compatible)
SAVE_DIR = Path.home() / "Documents" / "DentalDetection"
PHOTO_DIR = SAVE_DIR / "Photos"
VIDEO_DIR = SAVE_DIR / "Videos"
PHOTO_DIR.mkdir(parents=True, exist_ok=True)
VIDEO_DIR.mkdir(parents=True, exist_ok=True)

class RecordVideo(QtCore.QObject):
    image_data = QtCore.Signal(np.ndarray)
    recording_status = QtCore.Signal(bool)

    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.camera_port = camera_port
        self.camera = cv2.VideoCapture(self.camera_port, cv2.CAP_DSHOW)  # Windows optimized
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.zoom_factor = 1.0
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.timerEvent)
        self.timer.start(30)
        
        # 🎥 Video recording variables
        self.is_recording = False
        self.video_writer = None
        self.current_frame = None

    def timerEvent(self):
        ret, frame = self.camera.read()
        if ret:
            # Apply digital zoom
            if self.zoom_factor > 1.0:
                frame = self.apply_zoom(frame, self.zoom_factor)
            
            self.current_frame = frame.copy()
            self.image_data.emit(frame)
            
            # 🎥 Write frame if recording
            if self.is_recording and self.video_writer is not None:
                self.video_writer.write(frame)

    def apply_zoom(self, frame, zoom_factor):
        """Apply digital zoom by cropping and resizing"""
        h, w = frame.shape[:2]
        new_h, new_w = int(h / zoom_factor), int(w / zoom_factor)
        start_h = (h - new_h) // 2
        start_w = (w - new_w) // 2
        cropped = frame[start_h:start_h + new_h, start_w:start_w + new_w]
        zoomed = cv2.resize(cropped, (w, h), interpolation=cv2.INTER_LINEAR)
        return zoomed

    def set_zoom(self, zoom_factor):
        """Set zoom factor (1.0 = no zoom, 5.0 = 5x zoom)"""
        self.zoom_factor = max(1.0, min(zoom_factor, 5.0))

    def take_photo(self):
        """📸 Capture and save current frame"""
        if self.current_frame is not None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = PHOTO_DIR / f"dental_photo_{timestamp}.jpg"
            cv2.imwrite(str(filename), self.current_frame)
            return str(filename)
        return None

    def start_recording(self):
        """🎥 Start video recording"""
        if not self.is_recording:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = VIDEO_DIR / f"dental_video_{timestamp}.mp4"
            
            # Get frame dimensions
            h, w = self.current_frame.shape[:2] if self.current_frame is not None else (720, 1280)
            
            # Initialize video writer (H264 codec for Windows)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(str(filename), fourcc, 30.0, (w, h))
            
            self.is_recording = True
            self.recording_status.emit(True)
            return str(filename)
        return None

    def stop_recording(self):
        """⏹️ Stop video recording"""
        if self.is_recording:
            self.is_recording = False
            if self.video_writer is not None:
                self.video_writer.release()
                self.video_writer = None
            self.recording_status.emit(False)

    def __del__(self):
        """Clean up camera on exit"""
        if self.camera.isOpened():
            self.camera.release()

class TFLiteDetectionWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.image = QImage()
        self.detected_frame = None

    def detect_tflite(self, frame):
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(img_rgb, (640, 640)).astype(np.float32)
        input_tensor = np.expand_dims(resized, axis=0) / 255.0
        interpreter.set_tensor(input_details[0]['index'], input_tensor)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])[0]

        h, w, _ = frame.shape
        for det in output:
            x1, y1, x2, y2, score, class_id = det
            if score < 0.25:
                continue
            x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            label = f"{CLASS_NAMES[int(class_id)]}: {score:.2f}"
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
        return frame

    def image_data_slot(self, image_data):
        detected = self.detect_tflite(image_data)
        self.detected_frame = detected
        self.image = self.get_qimage(detected)
        self.update()

    def get_qimage(self, image):
        height, width, _ = image.shape
        return QImage(image.data, width, height, 3 * width, QImage.Format_BGR888)

    def paintEvent(self, event):
        painter = QtGui.QPainter(self)
        if not self.image.isNull():
            scaled = self.image.scaled(self.size(), QtCore.Qt.KeepAspectRatio,
                                       QtCore.Qt.SmoothTransformation)
            x = (self.width() - scaled.width()) // 2
            y = (self.height() - scaled.height()) // 2
            painter.drawImage(x, y, scaled)

class MainWidget(QtWidgets.QWidget):
    def __init__(self, camera_port=0, parent=None):
        super().__init__(parent)
        self.detector_widget = TFLiteDetectionWidget()
        self.video = RecordVideo(camera_port)
        self.video.image_data.connect(self.detector_widget.image_data_slot)
        self.video.recording_status.connect(self.update_recording_ui)

        # ✅ Zoom controls
        zoom_label = QtWidgets.QLabel("Zoom:")
        self.zoom_slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.zoom_slider.setMinimum(10)
        self.zoom_slider.setMaximum(50)
        self.zoom_slider.setValue(10)
        self.zoom_slider.setTickPosition(QtWidgets.QSlider.TicksBelow)
        self.zoom_slider.setTickInterval(5)
        self.zoom_slider.valueChanged.connect(self.update_zoom)

        self.zoom_value_label = QtWidgets.QLabel("1.0x")
        self.zoom_value_label.setFixedWidth(50)

        reset_button = QtWidgets.QPushButton("Reset Zoom")
        reset_button.clicked.connect(self.reset_zoom)

        # 📸 Capture controls
        self.photo_button = QtWidgets.QPushButton("📸 Take Photo")
        self.photo_button.setStyleSheet("QPushButton { background-color: #4CAF50; color: white; font-size: 14px; padding: 10px; }")
        self.photo_button.clicked.connect(self.take_photo)

        self.record_button = QtWidgets.QPushButton("🎥 Start Recording")
        self.record_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
        self.record_button.clicked.connect(self.toggle_recording)

        # 📁 Open folder button
        self.open_folder_button = QtWidgets.QPushButton("📁 Open Save Folder")
        self.open_folder_button.clicked.connect(self.open_save_folder)

        # Status label
        self.status_label = QtWidgets.QLabel(f"Ready | Saving to: {SAVE_DIR}")
        self.status_label.setAlignment(QtCore.Qt.AlignCenter)
        self.status_label.setStyleSheet("font-size: 11px; padding: 5px;")

        # Recording indicator
        self.recording_indicator = QtWidgets.QLabel("")
        self.recording_indicator.setAlignment(QtCore.Qt.AlignCenter)
        self.recording_indicator.setStyleSheet("font-size: 14px; color: red; font-weight: bold;")

        # ✅ Layout
        zoom_layout = QtWidgets.QHBoxLayout()
        zoom_layout.addWidget(zoom_label)
        zoom_layout.addWidget(self.zoom_slider)
        zoom_layout.addWidget(self.zoom_value_label)
        zoom_layout.addWidget(reset_button)

        button_layout = QtWidgets.QHBoxLayout()
        button_layout.addWidget(self.photo_button)
        button_layout.addWidget(self.record_button)
        button_layout.addWidget(self.open_folder_button)

        main_layout = QtWidgets.QVBoxLayout()
        main_layout.addWidget(self.detector_widget)
        main_layout.addWidget(self.recording_indicator)
        main_layout.addLayout(zoom_layout)
        main_layout.addLayout(button_layout)
        main_layout.addWidget(self.status_label)
        self.setLayout(main_layout)

    def update_zoom(self, value):
        """Update zoom when slider changes"""
        zoom_factor = value / 10.0
        self.video.set_zoom(zoom_factor)
        self.zoom_value_label.setText(f"{zoom_factor:.1f}x")

    def reset_zoom(self):
        """Reset zoom to 1.0x"""
        self.zoom_slider.setValue(10)

    def take_photo(self):
        """📸 Take a photo"""
        filename = self.video.take_photo()
        if filename:
            self.status_label.setText(f"✅ Photo saved: {Path(filename).name}")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(f"Ready | Saving to: {SAVE_DIR}"))
        else:
            self.status_label.setText("❌ Failed to capture photo")

    def toggle_recording(self):
        """🎥 Start/Stop video recording"""
        if not self.video.is_recording:
            filename = self.video.start_recording()
            if filename:
                self.status_label.setText(f"🎥 Recording: {Path(filename).name}")
        else:
            self.video.stop_recording()
            self.status_label.setText("✅ Recording saved")
            QtCore.QTimer.singleShot(3000, lambda: self.status_label.setText(f"Ready | Saving to: {SAVE_DIR}"))

    def open_save_folder(self):
        """📁 Open the save folder in file explorer"""
        if sys.platform == 'win32':
            os.startfile(SAVE_DIR)
        elif sys.platform == 'darwin':  # macOS
            os.system(f'open "{SAVE_DIR}"')
        else:  # Linux
            os.system(f'xdg-open "{SAVE_DIR}"')

    def update_recording_ui(self, is_recording):
        """Update UI based on recording status"""
        if is_recording:
            self.record_button.setText("⏹️ Stop Recording")
            self.record_button.setStyleSheet("QPushButton { background-color: #f44336; color: white; font-size: 14px; padding: 10px; }")
            self.recording_indicator.setText("● REC")
            self.photo_button.setEnabled(False)
        else:
            self.record_button.setText("🎥 Start Recording")
            self.record_button.setStyleSheet("QPushButton { background-color: #2196F3; color: white; font-size: 14px; padding: 10px; }")
            self.recording_indicator.setText("")
            self.photo_button.setEnabled(True)

def find_camera():
    """Find available camera (USB camera preferred)"""
    for i in range(5):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            if ret:
                print(f"Camera found at index {i}")
                return i
    return 0

if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    # Auto-detect camera
    camera_index = find_camera()

    main_window = QtWidgets.QMainWindow()
    main_widget = MainWidget(camera_port=camera_index)
    main_window.setCentralWidget(main_widget)
    main_window.setWindowTitle("🦷 Dental Detection - USB Camera")
    main_window.resize(900, 750)
    main_window.show()

    sys.exit(app.exec())
