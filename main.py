import sys
import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                            QHBoxLayout, QLabel, QLineEdit, QPushButton, 
                            QComboBox, QSpinBox, QSlider, QProgressBar, 
                            QFileDialog, QGroupBox, QGridLayout, 
                            QColorDialog, QMessageBox, QListWidget, QListWidgetItem,
                            QFrame, QSplitter, QCheckBox)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QSettings
from PyQt5.QtGui import QFont, QColor, QIcon

class VideoProcessor(QThread):
    progress = pyqtSignal(int, int, int)  # current_video, total_videos, video_progress
    current_file = pyqtSignal(str)
    finished = pyqtSignal(list)  # list of output paths
    error = pyqtSignal(str)
    analysis_progress = pyqtSignal(str)  # New signal for analysis updates

    def __init__(self, video_paths, brand_text, font_size, font_color, opacity, font_path=None, smart_positioning=True, analysis_frames=50):
        super().__init__()
        self.video_paths = video_paths
        self.brand_text = brand_text
        self.font_size = font_size
        self.font_color = font_color
        self.opacity = opacity
        self.font_path = font_path
        self.smart_positioning = smart_positioning
        self.analysis_frames = analysis_frames
        self.safe_zone_margins = {
            'left_right_percent': 11.1,
            'top_percent': 13.0,
            'bottom_percent': 19.8
        }
        # Cache optimal position for each video - now calculated from entire video
        self.cached_position = None

    def get_output_path(self, input_path):
        """Generate output path with PROSWIPE suffix in same directory."""
        path = Path(input_path)
        parent_dir = path.parent
        stem = path.stem
        suffix = path.suffix
        return str(parent_dir / f"{stem} PROSWIPE{suffix}")

    def detect_text_in_region(self, frame_region):
        """Comprehensive text detection using multiple advanced methods."""
        if frame_region.size == 0 or frame_region.shape[0] < 5 or frame_region.shape[1] < 5:
            return 0.0

        gray = cv2.cvtColor(frame_region, cv2.COLOR_BGR2GRAY)
        text_confidence = 0.0

        # Method 1: MSER (Maximally Stable Extremal Regions) - Best for text detection
        try:
            mser = cv2.MSER_create(
                _delta=5,
                _min_area=10,
                _max_area=gray.shape[0] * gray.shape[1] // 2,
                _max_variation=0.25,
                _min_diversity=0.2
            )
            regions, _ = mser.detectRegions(gray)
            mser_score = min(len(regions) * 0.5, 10.0)
            text_confidence += mser_score
        except:
            pass

        # Method 2: Connected Components Analysis for text-like structures
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh, connectivity=8)
        
        text_like_components = 0
        for i in range(1, num_labels):  # Skip background
            area = stats[i, cv2.CC_STAT_AREA]
            width = stats[i, cv2.CC_STAT_WIDTH]
            height = stats[i, cv2.CC_STAT_HEIGHT]
            if area > 20 and width > 5 and height > 5:
                aspect_ratio = width / height if height > 0 else 0
                if 0.1 < aspect_ratio < 10.0:
                    text_like_components += 1
        text_confidence += text_like_components * 0.4

        # Method 3: Edge-based analysis
        edges = cv2.Canny(gray, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        stroke_like_contours = 0
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 20:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h if h > 0 else 0
                if 0.2 < aspect_ratio < 8.0 and w > 3 and h > 3:
                    stroke_like_contours += 1
        text_confidence += stroke_like_contours * 0.3

        # Method 4: Horizontal line density
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
        horizontal_pixels = np.sum(horizontal_lines > 0)
        horizontal_density = horizontal_pixels / (horizontal_lines.shape[0] * horizontal_lines.shape[1])
        text_confidence += horizontal_density * 15.0

        # Method 5: Vertical transition analysis
        vertical_transitions = 0
        for col in range(gray.shape[1]):
            column = gray[:, col]
            transitions = np.sum(np.abs(np.diff(column.astype(np.float32))) > 30)
            vertical_transitions += transitions
        
        if gray.shape[1] > 0:
            transition_density = vertical_transitions / (gray.shape[0] * gray.shape[1])
            text_confidence += transition_density * 20.0

        # Method 6: Local Binary Pattern for text texture
        rows, cols = gray.shape
        if rows > 4 and cols > 4:
            lbp_like_score = 0
            for i in range(2, rows-2):
                for j in range(2, cols-2):
                    center = gray[i, j]
                    neighbors = [
                        gray[i-1, j-1], gray[i-1, j], gray[i-1, j+1],
                        gray[i, j+1], gray[i+1, j+1], gray[i+1, j],
                        gray[i+1, j-1], gray[i, j-1]
                    ]
                    greater_neighbors = sum(1 for n in neighbors if n > center + 10)
                    if 2 <= greater_neighbors <= 6:
                        lbp_like_score += 1
            lbp_density = lbp_like_score / (rows * cols)
            text_confidence += lbp_density * 25.0

        return min(text_confidence, 50.0)  # Cap the score

    def analyze_entire_video_for_optimal_position(self, cap, text_width, text_height):
        """Analyze the ENTIRE video to find the globally optimal watermark position."""
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        # Calculate safe zone boundaries
        left_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        right_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        top_margin = int(frame_height * self.safe_zone_margins['top_percent'] / 100)
        bottom_margin = int(frame_height * self.safe_zone_margins['bottom_percent'] / 100)

        safe_x_start = left_margin
        safe_x_end = frame_width - right_margin
        safe_y_start = top_margin
        safe_y_end = frame_height - bottom_margin

        padding = 20
        available_width = safe_x_end - safe_x_start - text_width - (2 * padding)
        available_height = safe_y_end - safe_y_start - text_height - (2 * padding)

        if available_width <= 0 or available_height <= 0:
            return safe_x_start + (safe_x_end - safe_x_start) // 2 - text_width // 2, \
                   safe_y_start + (safe_y_end - safe_y_start) // 2 - text_height // 2

        # Create a grid of potential positions
        grid_x = min(12, available_width // 25)  # Reasonable grid size
        grid_y = min(8, available_height // 25)
        if grid_x <= 0: grid_x = 1
        if grid_y <= 0: grid_y = 1

        # Initialize position scores
        position_scores = {}
        for i in range(grid_x):
            for j in range(grid_y):
                position_scores[(i, j)] = []

        # Sample frames more intelligently - focus on different parts of video
        sample_interval = max(1, total_frames // self.analysis_frames)  # Sample based on user choice
        sample_frames = list(range(0, total_frames, sample_interval))
        
        self.analysis_progress.emit("🔍 Analyzing entire video for optimal placement...")
        
        frames_analyzed = 0
        for frame_idx in sample_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            frames_analyzed += 1
            
            # Update progress
            analysis_percent = (frames_analyzed / len(sample_frames)) * 100
            self.analysis_progress.emit(f"🔍 Analyzing frame {frames_analyzed}/{len(sample_frames)} ({analysis_percent:.1f}%)")

            # Test each position on this frame
            for i in range(grid_x):
                for j in range(grid_y):
                    # Calculate position
                    if grid_x == 1:
                        x_offset = 0
                    else:
                        x_offset = int(i * available_width / (grid_x - 1))
                    
                    if grid_y == 1:
                        y_offset = 0
                    else:
                        y_offset = int(j * available_height / (grid_y - 1))

                    test_x = safe_x_start + padding + x_offset
                    test_y = safe_y_start + padding + y_offset

                    # Ensure within bounds
                    test_x = max(safe_x_start + padding, min(test_x, safe_x_end - text_width - padding))
                    test_y = max(safe_y_start + padding, min(test_y, safe_y_end - text_height - padding))

                    # Extract region for analysis
                    analysis_margin = 15
                    analysis_x1 = max(0, test_x - analysis_margin)
                    analysis_y1 = max(0, test_y - analysis_margin)
                    analysis_x2 = min(frame_width, test_x + text_width + analysis_margin)
                    analysis_y2 = min(frame_height, test_y + text_height + analysis_margin)

                    text_region = frame[analysis_y1:analysis_y2, analysis_x1:analysis_x2]
                    if text_region.size > 0:
                        text_score = self.detect_text_in_region(text_region)
                        position_scores[(i, j)].append(text_score)

        # Calculate the best position based on entire video analysis
        best_position = None
        lowest_average_score = float('inf')

        for (i, j), scores in position_scores.items():
            if scores:  # Make sure we have scores for this position
                # Use average score across all analyzed frames
                avg_score = sum(scores) / len(scores)
                # Also consider the maximum score (worst case scenario)
                max_score = max(scores)
                # Combined metric: heavily weight the average, but penalize high peaks
                combined_score = avg_score * 2.0 + max_score * 0.5
                
                if combined_score < lowest_average_score:
                    lowest_average_score = combined_score
                    
                    # Calculate actual pixel position
                    if grid_x == 1:
                        x_offset = 0
                    else:
                        x_offset = int(i * available_width / (grid_x - 1))
                    
                    if grid_y == 1:
                        y_offset = 0
                    else:
                        y_offset = int(j * available_height / (grid_y - 1))

                    best_x = safe_x_start + padding + x_offset
                    best_y = safe_y_start + padding + y_offset
                    best_position = (best_x, best_y)

        if best_position is None:
            # Fallback position
            best_position = (safe_x_start + padding, safe_y_start + padding)

        self.analysis_progress.emit(f"✅ Analysis complete! Found optimal position with score: {lowest_average_score:.2f}")
        
        # Reset video to beginning
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        return best_position

    def get_text_position(self, frame, frame_width, frame_height, text_width, text_height):
        """Get the optimal text position - now uses cached position from full video analysis."""
        if not self.smart_positioning:
            return self.get_text_position_fallback(frame_width, frame_height, text_width, text_height)

        # Return the cached position calculated from entire video
        if self.cached_position:
            return self.cached_position
        
        # This should not happen if analyze_entire_video_for_optimal_position was called
        # Fallback to simple analysis
        return self.find_best_position_globally(frame, text_width, text_height)

    def find_best_position_globally(self, frame, text_width, text_height):
        """Find the best position for a single frame (fallback method)."""
        frame_height, frame_width = frame.shape[:2]
        
        left_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        right_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        top_margin = int(frame_height * self.safe_zone_margins['top_percent'] / 100)
        bottom_margin = int(frame_height * self.safe_zone_margins['bottom_percent'] / 100)

        safe_x_start = left_margin
        safe_x_end = frame_width - right_margin
        safe_y_start = top_margin
        safe_y_end = frame_height - bottom_margin

        padding = 20
        available_width = safe_x_end - safe_x_start - text_width - (2 * padding)
        available_height = safe_y_end - safe_y_start - text_height - (2 * padding)

        if available_width <= 0 or available_height <= 0:
            return safe_x_start + (safe_x_end - safe_x_start) // 2 - text_width // 2, \
                   safe_y_start + (safe_y_end - safe_y_start) // 2 - text_height // 2

        grid_x = min(15, available_width // 20)
        grid_y = min(10, available_height // 20)
        if grid_x <= 0: grid_x = 1
        if grid_y <= 0: grid_y = 1

        best_position = None
        lowest_text_score = float('inf')

        for i in range(grid_x):
            for j in range(grid_y):
                if grid_x == 1:
                    x_offset = 0
                else:
                    x_offset = int(i * available_width / (grid_x - 1))
                
                if grid_y == 1:
                    y_offset = 0
                else:
                    y_offset = int(j * available_height / (grid_y - 1))

                test_x = safe_x_start + padding + x_offset
                test_y = safe_y_start + padding + y_offset

                test_x = max(safe_x_start + padding, min(test_x, safe_x_end - text_width - padding))
                test_y = max(safe_y_start + padding, min(test_y, safe_y_end - text_height - padding))

                analysis_margin = 15
                analysis_x1 = max(0, test_x - analysis_margin)
                analysis_y1 = max(0, test_y - analysis_margin)
                analysis_x2 = min(frame_width, test_x + text_width + analysis_margin)
                analysis_y2 = min(frame_height, test_y + text_height + analysis_margin)

                text_region = frame[analysis_y1:analysis_y2, analysis_x1:analysis_x2]
                if text_region.size > 0:
                    text_score = self.detect_text_in_region(text_region)
                    gray_region = cv2.cvtColor(text_region, cv2.COLOR_BGR2GRAY)
                    complexity = np.std(gray_region) / 255.0
                    combined_score = text_score * 10.0 + complexity * 1.0

                    if combined_score < lowest_text_score:
                        lowest_text_score = combined_score
                        best_position = (test_x, test_y)

        return best_position if best_position else (safe_x_start + padding, safe_y_start + padding)

    def get_text_position_fallback(self, frame_width, frame_height, text_width, text_height):
        """Simple fallback positioning when smart positioning is disabled."""
        left_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        right_margin = int(frame_width * self.safe_zone_margins['left_right_percent'] / 100)
        top_margin = int(frame_height * self.safe_zone_margins['top_percent'] / 100)
        bottom_margin = int(frame_height * self.safe_zone_margins['bottom_percent'] / 100)

        x = frame_width - right_margin - text_width - 20
        y = frame_height - bottom_margin - text_height - 20

        x = max(left_margin + 20, x)
        y = max(top_margin + 20, y)
        return x, y

    def create_text_overlay(self, frame):
        pil_frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        overlay = Image.new('RGBA', pil_frame.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)

        try:
            if self.font_path and os.path.exists(self.font_path):
                font = ImageFont.truetype(self.font_path, self.font_size)
            else:
                font = ImageFont.truetype("arial.ttf", self.font_size)
        except:
            font = ImageFont.load_default()

        bbox = draw.textbbox((0, 0), self.brand_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]

        x, y = self.get_text_position(frame, pil_frame.width, pil_frame.height, text_width, text_height)

        outline_color = (0, 0, 0, int(255 * self.opacity))
        text_color = (*self.font_color, int(255 * self.opacity))

        outline_width = 4
        for dx in range(-outline_width, outline_width + 1):
            for dy in range(-outline_width, outline_width + 1):
                if dx != 0 or dy != 0:
                    draw.text((x + dx, y + dy), self.brand_text, font=font, fill=outline_color)

        draw.text((x, y), self.brand_text, font=font, fill=text_color)

        result = Image.alpha_composite(pil_frame.convert('RGBA'), overlay)
        return cv2.cvtColor(np.array(result.convert('RGB')), cv2.COLOR_RGB2BGR)

    def run(self):
        try:
            import tempfile
            from moviepy.editor import VideoFileClip, AudioFileClip
            
            output_paths = []
            total_videos = len(self.video_paths)

            for video_idx, input_path in enumerate(self.video_paths):
                self.current_file.emit(f"Processing: {os.path.basename(input_path)}")
                self.cached_position = None  # Reset cache for each video

                cap = cv2.VideoCapture(input_path)
                if not cap.isOpened():
                    self.error.emit(f"Could not open: {input_path}")
                    continue

                fps = cap.get(cv2.CAP_PROP_FPS)
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                # If smart positioning is enabled, analyze the entire video first
                if self.smart_positioning:
                    # Get text dimensions for analysis
                    try:
                        if self.font_path and os.path.exists(self.font_path):
                            font = ImageFont.truetype(self.font_path, self.font_size)
                        else:
                            font = ImageFont.truetype("arial.ttf", self.font_size)
                    except:
                        font = ImageFont.load_default()
                    
                    # Create temporary image to get text dimensions
                    temp_img = Image.new('RGB', (100, 100))
                    temp_draw = ImageDraw.Draw(temp_img)
                    bbox = temp_draw.textbbox((0, 0), self.brand_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]
                    
                    # Analyze entire video to find optimal position
                    self.cached_position = self.analyze_entire_video_for_optimal_position(cap, text_width, text_height)

                # Create temporary silent video
                temp_fd, temp_path = tempfile.mkstemp(suffix=".mp4")
                os.close(temp_fd)

                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out = cv2.VideoWriter(temp_path, fourcc, fps, (width, height))

                frame_count = 0
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset to beginning

                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    branded_frame = self.create_text_overlay(frame)
                    out.write(branded_frame)
                    frame_count += 1
                    
                    video_progress = int((frame_count / total_frames) * 100)
                    self.progress.emit(video_idx + 1, total_videos, video_progress)

                cap.release()
                out.release()

                # Final output path
                output_path = self.get_output_path(input_path)

                # Mux audio back in with MoviePy
                video = VideoFileClip(temp_path)
                orig = VideoFileClip(input_path)
                
                if orig.audio:
                    video = video.set_audio(orig.audio)

                video.write_videofile(
                    output_path,
                    codec="libx264",
                    audio_codec="aac",
                    temp_audiofile="temp-audio.m4a",
                    remove_temp=True,
                    threads=4,
                    preset="medium",
                    verbose=False,
                    logger=None,
                )

                output_paths.append(output_path)
                os.remove(temp_path)  # cleanup

            cv2.destroyAllWindows()
            self.finished.emit(output_paths)

        except Exception as e:
            self.error.emit(str(e))


class BrandWatermarkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.video_paths = []
        self.font_path = ""
        self.font_color = (255, 255, 0)
        
        # Initialize settings
        self.settings = QSettings("VideoWatermark", "BrandTool")
        self.initUI()
        self.load_settings()

    def initUI(self):
        self.setWindowTitle('Video Brand Watermark Tool - Enhanced')
        self.setGeometry(100, 100, 950, 700)
        self.setMinimumSize(850, 650)
        # Set window icon
        if getattr(sys, 'frozen', False):
            # Running as PyInstaller bundle
            icon_path = os.path.join(sys._MEIPASS, 'lg.ico')
        else:
            # Running as script
            icon_path = 'lg.ico'

        if os.path.exists(icon_path):
            self.setWindowIcon(QIcon(icon_path))
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        layout = QVBoxLayout(central_widget)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title_label = QLabel('Video Brand Watermark Tool - Enhanced Full Video Analysis')
        title_label.setAlignment(Qt.AlignCenter)
        title_font = QFont()
        title_font.setPointSize(16)
        title_font.setBold(True)
        title_label.setFont(title_font)
        layout.addWidget(title_label)

        # Splitter for main content
        splitter = QSplitter(Qt.Horizontal)

        # Left side - File management
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)

        # File selection group
        file_group = QGroupBox("Video Files")
        file_layout = QVBoxLayout(file_group)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMinimumHeight(200)
        file_layout.addWidget(self.file_list)

        # File buttons
        file_btn_layout = QHBoxLayout()
        self.add_files_btn = QPushButton("Add Videos")
        self.add_files_btn.clicked.connect(self.add_videos)
        file_btn_layout.addWidget(self.add_files_btn)

        self.clear_files_btn = QPushButton("Clear All")
        self.clear_files_btn.clicked.connect(self.clear_videos)
        file_btn_layout.addWidget(self.clear_files_btn)

        file_layout.addLayout(file_btn_layout)
        left_layout.addWidget(file_group)

        # Right side - Brand settings
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)

        brand_group = QGroupBox("Brand Settings")
        brand_layout = QGridLayout(brand_group)
        brand_layout.setSpacing(10)

        row = 0

        # Brand text
        brand_layout.addWidget(QLabel("Brand Text:"), row, 0)
        self.brand_text = QLineEdit("PROSWIPE")
        brand_layout.addWidget(self.brand_text, row, 1, 1, 2)
        row += 1

        # Font size
        brand_layout.addWidget(QLabel("Font Size:"), row, 0)
        self.font_size = QSpinBox()
        self.font_size.setRange(20, 200)
        self.font_size.setValue(60)
        self.font_size.setSuffix(" px")
        brand_layout.addWidget(self.font_size, row, 1)
        row += 1

        # Smart positioning checkbox
        self.smart_positioning_cb = QCheckBox("Intelligent Text Avoidance")
        self.smart_positioning_cb.setChecked(True)
        self.smart_positioning_cb.setToolTip("Enable intelligent text detection and avoidance")
        brand_layout.addWidget(self.smart_positioning_cb, row, 0, 1, 3)
        row += 1

        # Frame analysis mode selection
        brand_layout.addWidget(QLabel("Analysis Depth:"), row, 0)
        self.analysis_mode = QComboBox()
        self.analysis_mode.addItems(["Quick (4 frames)", "Deep (50 frames)"])
        self.analysis_mode.setCurrentIndex(1)  # Default to Deep
        self.analysis_mode.setToolTip("Quick: Fast processing, good results\nDeep: Slower but optimal results")
        brand_layout.addWidget(self.analysis_mode, row, 1, 1, 2)
        row += 1

        # Opacity
        brand_layout.addWidget(QLabel("Opacity:"), row, 0)
        self.opacity_slider = QSlider(Qt.Horizontal)
        self.opacity_slider.setRange(10, 100)
        self.opacity_slider.setValue(80)
        self.opacity_slider.valueChanged.connect(self.update_opacity_label)
        brand_layout.addWidget(self.opacity_slider, row, 1)

        self.opacity_label = QLabel("80%")
        brand_layout.addWidget(self.opacity_label, row, 2)
        row += 1

        # Text color
        brand_layout.addWidget(QLabel("Text Color:"), row, 0)
        self.color_btn = QPushButton("Choose Color")
        self.color_btn.clicked.connect(self.choose_color)
        brand_layout.addWidget(self.color_btn, row, 1, 1, 2)
        row += 1

        # Font file
        brand_layout.addWidget(QLabel("Font File:"), row, 0)
        self.font_btn = QPushButton("Browse Font")
        self.font_btn.clicked.connect(self.browse_font_file)
        brand_layout.addWidget(self.font_btn, row, 1, 1, 2)

        right_layout.addWidget(brand_group)

        # Enhanced explanation
        explanation_group = QGroupBox("🚀 Enhanced Full Video Analysis System")
        explanation_layout = QVBoxLayout(explanation_group)
        
        explanation_text = QLabel(
            "NEW: Analyzes the ENTIRE video before processing!\n"
            "• Samples ~50 frames throughout the complete video\n"
            "• Tests each position across ALL sampled frames\n"
            "• Finds the position with lowest text interference globally\n"
            "• Ensures consistent placement that avoids text throughout the video\n"
            "• Uses advanced averaging to handle dynamic text content"
        )
        explanation_text.setWordWrap(True)
        explanation_text.setStyleSheet("QLabel { color: #2c5aa0; font-size: 11px; background-color: #f0fff0; padding: 12px; border-radius: 6px; border-left: 4px solid #28a745; }")
        explanation_layout.addWidget(explanation_text)
        right_layout.addWidget(explanation_group)

        # Add widgets to splitter
        splitter.addWidget(left_widget)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter)

        # Processing section
        process_group = QGroupBox("Processing")
        process_layout = QVBoxLayout(process_group)

        # Process button
        self.process_btn = QPushButton("🚀 Start Full Video Analysis")
        process_font = QFont()
        process_font.setPointSize(12)
        process_font.setBold(True)
        self.process_btn.setFont(process_font)
        self.process_btn.setMinimumHeight(45)
        self.process_btn.clicked.connect(self.process_videos)
        process_layout.addWidget(self.process_btn)

        # Current file label
        self.current_file_label = QLabel("")
        self.current_file_label.setAlignment(Qt.AlignCenter)
        self.current_file_label.setVisible(False)
        process_layout.addWidget(self.current_file_label)

        # Analysis progress label (new)
        self.analysis_label = QLabel("")
        self.analysis_label.setAlignment(Qt.AlignCenter)
        self.analysis_label.setStyleSheet("QLabel { color: #28a745; font-weight: bold; }")
        self.analysis_label.setVisible(False)
        process_layout.addWidget(self.analysis_label)

        # Progress bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimumHeight(25)
        self.progress_bar.setVisible(False)
        process_layout.addWidget(self.progress_bar)

        layout.addWidget(process_group)

        # Status bar
        self.statusBar().showMessage("Ready - Enhanced with full video analysis")

    def load_settings(self):
        """Load saved settings from QSettings"""
        brand_text = self.settings.value("brand_text", "PROSWIPE")
        self.brand_text.setText(brand_text)

        font_size = self.settings.value("font_size", 60, type=int)
        self.font_size.setValue(font_size)

        smart_positioning = self.settings.value("smart_positioning", True, type=bool)
        self.smart_positioning_cb.setChecked(smart_positioning)

        analysis_mode = self.settings.value("analysis_mode", 1, type=int)  # Default to Deep
        self.analysis_mode.setCurrentIndex(analysis_mode)

        opacity = self.settings.value("opacity", 80, type=int)
        self.opacity_slider.setValue(opacity)
        self.update_opacity_label(opacity)

        color_r = self.settings.value("text_color_r", 255, type=int)
        color_g = self.settings.value("text_color_g", 255, type=int)
        color_b = self.settings.value("text_color_b", 0, type=int)
        self.font_color = (color_r, color_g, color_b)
        self.update_color_button_text()

        font_path = self.settings.value("font_path", "")
        if font_path and os.path.exists(font_path):
            self.font_path = font_path
            self.font_btn.setText(f"Font: {os.path.basename(font_path)}")
        else:
            self.font_path = ""
            self.font_btn.setText("Browse Font")

    def save_settings(self):
        """Save current settings to QSettings"""
        self.settings.setValue("brand_text", self.brand_text.text())
        self.settings.setValue("font_size", self.font_size.value())
        self.settings.setValue("smart_positioning", self.smart_positioning_cb.isChecked())
        # Save analysis mode
        self.settings.setValue("analysis_mode", self.analysis_mode.currentIndex())
        self.settings.setValue("opacity", self.opacity_slider.value())
        self.settings.setValue("text_color_r", self.font_color[0])
        self.settings.setValue("text_color_g", self.font_color[1])
        self.settings.setValue("text_color_b", self.font_color[2])
        self.settings.setValue("font_path", self.font_path)

    def update_color_button_text(self):
        """Update the color button text to show current color"""
        r, g, b = self.font_color
        self.color_btn.setText(f"Color: RGB({r}, {g}, {b})")

    def add_videos(self):
        last_dir = self.settings.value("last_directory", "")
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", last_dir, 
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv *.flv);;All Files (*)"
        )
        if files:
            first_file_dir = os.path.dirname(files[0])
            self.settings.setValue("last_directory", first_file_dir)
            for file_path in files:
                if file_path not in self.video_paths:
                    self.video_paths.append(file_path)
                    self.file_list.addItem(os.path.basename(file_path))
        self.statusBar().showMessage(f"{len(self.video_paths)} video(s) loaded.")

    def clear_videos(self):
        self.video_paths.clear()
        self.file_list.clear()
        self.statusBar().showMessage("Video list cleared")

    def choose_color(self):
        current_color = QColor(*self.font_color)
        color = QColorDialog.getColor(current_color)
        if color.isValid():
            self.font_color = (color.red(), color.green(), color.blue())
            self.update_color_button_text()
            self.save_settings()

    def browse_font_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Font File", "", 
            "Font Files (*.ttf *.otf);;All Files (*)"
        )
        if file_path:
            self.font_path = file_path
            self.font_btn.setText(f"Font: {os.path.basename(file_path)}")
            self.save_settings()

    def update_opacity_label(self, value):
        self.opacity_label.setText(f"{value}%")

    def process_videos(self):
        if not self.video_paths:
            QMessageBox.warning(self, "No Videos", "Please add video files to process.")
            return

        if not self.brand_text.text().strip():
            QMessageBox.warning(self, "No Brand Text", "Please enter brand text.")
            return

        # Show analysis time warning for full video analysis
        if self.smart_positioning_cb.isChecked():
            frames_count = 4 if self.analysis_mode.currentIndex() == 0 else 50
            mode_name = "Quick" if self.analysis_mode.currentIndex() == 0 else "Deep"
            
            reply = QMessageBox.question(
                self, f"{mode_name} Video Analysis", 
                f"{mode_name} video analysis selected.\n\n"
                f"The system will:\n"
                f"• Analyze ~{frames_count} frames from each complete video\n"
                f"• Test multiple positions across all frames\n"
                f"• Find the globally optimal position\n\n"
                f"{'Quick mode: Faster processing' if frames_count == 4 else 'Deep mode: Longer processing but better results'}. Continue?",
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.Yes
            )
            if reply == QMessageBox.No:
                return

        self.save_settings()

        # Update UI for processing
        self.process_btn.setEnabled(False)
        self.process_btn.setText("🔍 Analyzing Videos...")
        self.current_file_label.setVisible(True)
        self.analysis_label.setVisible(True)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        # Start processing
        # Start processing
        self.processor = VideoProcessor(
            video_paths=self.video_paths.copy(),
            brand_text=self.brand_text.text(),
            font_size=self.font_size.value(),
            font_color=self.font_color,
            opacity=self.opacity_slider.value() / 100.0,
            font_path=self.font_path if self.font_path else None,
            smart_positioning=self.smart_positioning_cb.isChecked(),
            analysis_frames=4 if self.analysis_mode.currentIndex() == 0 else 50
        )

        self.processor.progress.connect(self.update_progress)
        self.processor.current_file.connect(self.update_current_file)
        self.processor.analysis_progress.connect(self.update_analysis_progress)  # New connection
        self.processor.finished.connect(self.processing_finished)
        self.processor.error.connect(self.processing_error)
        self.processor.start()

    def update_progress(self, current_video, total_videos, video_progress):
        overall_progress = ((current_video - 1) / total_videos) * 100 + (video_progress / total_videos)
        self.progress_bar.setValue(int(overall_progress))
        self.progress_bar.setFormat(f"Video {current_video}/{total_videos} - {video_progress}%")

    def update_current_file(self, filename):
        self.current_file_label.setText(filename)
        self.statusBar().showMessage(f"Processing: {filename}")

    def update_analysis_progress(self, message):
        """New method to update analysis progress"""
        self.analysis_label.setText(message)
        self.statusBar().showMessage(message)

    def processing_finished(self, output_paths):
        analysis_method = "full video analysis" if self.smart_positioning_cb.isChecked() else "simple positioning"
        QMessageBox.information(
            self, "🎉 Success!", 
            f"Successfully processed {len(output_paths)} videos using {analysis_method}!\n\n"
            f"📁 All files saved with 'PROSWIPE' suffix\n"
            f"🎯 Each video was analyzed completely for optimal watermark placement!"
        )
        self.reset_processing_ui()
        self.statusBar().showMessage(f"✅ Completed - {len(output_paths)} videos processed with {analysis_method}.")

    def processing_error(self, error_message):
        QMessageBox.critical(self, "❌ Error", f"Processing failed: {error_message}")
        self.reset_processing_ui()
        self.statusBar().showMessage("❌ Processing failed")

    def reset_processing_ui(self):
        self.process_btn.setEnabled(True)
        self.process_btn.setText("🚀 Start Full Video Analysis")
        self.current_file_label.setVisible(False)
        self.analysis_label.setVisible(False)
        self.progress_bar.setVisible(False)

    def closeEvent(self, event):
        """Save settings when the application is closed"""
        self.save_settings()
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Video Brand Watermark Tool - Enhanced")
    app.setApplicationVersion("4.0")
    
    window = BrandWatermarkGUI()
    window.show()
    
    sys.exit(app.exec_())


if __name__ == '__main__':
    try:
        import cv2
        import PIL
        from PyQt5.QtWidgets import QApplication
    except ImportError as e:
        print(f"Missing required library: {e}")
        print("Please install: pip install opencv-python pillow PyQt5 moviepy")
        sys.exit(1)
    
    main()