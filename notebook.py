#!/usr/bin/env python
# coding: utf-8

# # Bird's Eye View Calibration with YOLO Detection and DeepSORT Tracking
# 
# This notebook combines:
# 1. Bird's Eye View (BEV) calibration
# 2. YOLO object detection
# 3. DeepSORT tracking
# 
# ## Setup and Installation

# Install required packages
!pip install ultralytics
!pip install opencv-python
!pip install numpy
!pip install supervision
!pip install deep-sort-realtime

# Clone the DeepSORT repository if needed
!git clone https://github.com/nwojke/deep_sort.git

# Import necessary libraries
import os
import cv2
import numpy as np
import argparse
import torch
import supervision as sv
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ## Bird's Eye View Calibration Class
# Adapted from the original BEV Calibration toolkit

class BirdsEyeView:
    def __init__(self, image, coordinates, size=None):
        self.original = image.copy()
        self.image = image
        self.c, self.r = image.shape[0:2]
        if size:
            self.bc, self.br = size
        else:
            self.bc, self.br = self.c, self.r
        pst2 = np.float32(coordinates)
        pst1 = np.float32([[0,0], [self.r,0], [0,self.c], [self.r,self.c]])
        self.transferI2B = cv2.getPerspectiveTransform(pst1, pst2)
        self.transferB2I = cv2.getPerspectiveTransform(pst2, pst1)
        self.bird = self.img2bird()
        
    def img2bird(self):
        self.bird = cv2.warpPerspective(self.image, self.transferI2B, (self.br, self.bc))
        return self.bird
    
    def bird2img(self):
        self.image = cv2.warpPerspective(self.bird, self.transferB2I, (self.r, self.c))
        return self.image
    
    def convrt2Bird(self, img):
        return cv2.warpPerspective(img, self.transferI2B, (self.bird.shape[1], self.bird.shape[0]))
    
    def convrt2Image(self, bird):
        return cv2.warpPerspective(bird, self.transferB2I, (self.image.shape[1], self.image.shape[0]))
    
    def projection_on_bird(self, p, float_type=False):
        M = self.transferI2B
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)
    
    def projection_on_image(self, p, float_type=False):
        M = self.transferB2I
        px = (M[0][0]*p[0] + M[0][1]*p[1] + M[0][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        py = (M[1][0]*p[0] + M[1][1]*p[1] + M[1][2]) / ((M[2][0]*p[0] + M[2][1]*p[1] + M[2][2]))
        if float_type: return px, py
        return int(px), int(py)

# ## Helper Functions for BEV Calibration

def calcBackground(video_path, reduce=2, num_frames=100):
    """Extract background from video by averaging frames"""
    cap = cv2.VideoCapture(video_path)
    _, f = cap.read()
    f = cv2.resize(f, (f.shape[1]//reduce, f.shape[0]//reduce))
    img_bkgd = np.float32(f)
    
    frame_count = 0
    while frame_count < num_frames:
        ret, f = cap.read()
        if not ret: break
        f = cv2.resize(f, (f.shape[1]//reduce, f.shape[0]//reduce))
        cv2.accumulateWeighted(f, img_bkgd, 0.01)
        frame_count += 1
    
    background = cv2.convertScaleAbs(img_bkgd)
    cap.release()
    return background

def getROI(image):
    """Interactive ROI selection"""
    roi_coords = []
    roi = None
    
    def mouse_callback(event, x, y, flags, param):
        nonlocal roi_coords, roi
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(roi_coords) < 2:
                roi_coords.append((x, y))
                if len(roi_coords) == 2:
                    x1, y1 = roi_coords[0]
                    x2, y2 = roi_coords[1]
                    roi = image[min(y1,y2):max(y1,y2), min(x1,x2):max(x1,x2)]
    
    cv2.namedWindow("Select ROI")
    cv2.setMouseCallback("Select ROI", mouse_callback)
    
    while True:
        img_display = image.copy()
        if len(roi_coords) == 1:
            cv2.circle(img_display, roi_coords[0], 5, (0, 255, 0), -1)
        elif len(roi_coords) == 2:
            x1, y1 = roi_coords[0]
            x2, y2 = roi_coords[1]
            cv2.rectangle(img_display, (min(x1,x2), min(y1,y2)), (max(x1,x2), max(y1,y2)), (0, 255, 0), 2)
        
        cv2.imshow("Select ROI", img_display)
        key = cv2.waitKey(1)
        if key == 32 and len(roi_coords) == 2:  # SPACE key
            break
    
    cv2.destroyAllWindows()
    if len(roi_coords) == 2:
        x1, y1 = roi_coords[0]
        x2, y2 = roi_coords[1]
        roi_coords = [[min(x1,x2), min(y1,y2)], [max(x1,x2), max(y1,y2)]]
        return roi, roi_coords
    return None, None

def selectCalibrationPoints(image, num_points=4):
    """Select calibration points for BEV transformation"""
    points = []
    
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < num_points:
                points.append((x, y))
    
    cv2.namedWindow("Select Calibration Points")
    cv2.setMouseCallback("Select Calibration Points", mouse_callback)
    
    while True:
        img_display = image.copy()
        for i, point in enumerate(points):
            cv2.circle(img_display, point, 5, (0, 255, 0), -1)
            cv2.putText(img_display, str(i+1), (point[0]+10, point[1]+10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        cv2.imshow("Select Calibration Points", img_display)
        key = cv2.waitKey(1)
        if key == 32 and len(points) == num_points:  # SPACE key
            break
    
    cv2.destroyAllWindows()
    return points

def manualCalibration(image):
    """Manual calibration by selecting 4 points that form a rectangle in real world"""
    points = selectCalibrationPoints(image, 4)
    
    # Calculate width and height
    width_a = np.sqrt(((points[1][0] - points[0][0]) ** 2) + ((points[1][1] - points[0][1]) ** 2))
    width_b = np.sqrt(((points[3][0] - points[2][0]) ** 2) + ((points[3][1] - points[2][1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    
    height_a = np.sqrt(((points[1][0] - points[3][0]) ** 2) + ((points[1][1] - points[3][1]) ** 2))
    height_b = np.sqrt(((points[0][0] - points[2][0]) ** 2) + ((points[0][1] - points[2][1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    
    # Create destination points for perspective transform
    dst_points = np.float32([[0,0], [max_width,0], [0,max_height], [max_width,max_height]])
    src_points = np.float32(points)
    
    # Get perspective transform
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    
    # Project corners to get BEV coordinates
    h, w = image.shape[:2]
    corners = [(0,0), (w,0), (0,h), (w,h)]
    bev_coords = []
    
    for corner in corners:
        px = (M[0][0]*corner[0] + M[0][1]*corner[1] + M[0][2]) / ((M[2][0]*corner[0] + M[2][1]*corner[1] + M[2][2]))
        py = (M[1][0]*corner[0] + M[1][1]*corner[1] + M[1][2]) / ((M[2][0]*corner[0] + M[2][1]*corner[1] + M[2][2]))
        bev_coords.append([int(px), int(py)])
    
    # Create BEV transform
    bev = BirdsEyeView(image, bev_coords, size=(max_height, max_width))
    
    return bev_coords, (max_height, max_width)

# ## YOLO and DeepSORT Integration

class ObjectTracker:
    def __init__(self, yolo_model="yolov8n.pt", confidence=0.3, classes=None):
        # Initialize YOLO model
        self.model = YOLO(yolo_model)
        self.confidence = confidence
        self.classes = classes
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(max_age=30)
        
        # Initialize BEV transformer
        self.bev_transformer = None
        self.bev_coords = None
        self.bev_size = None
        self.pixel_unit = (1, 1)  # Default pixel to meter ratio
        
    def set_bev_calibration(self, bev_coords, bev_size, pixel_unit=(1, 1)):
        """Set BEV calibration parameters"""
        self.bev_coords = bev_coords
        self.bev_size = bev_size
        self.pixel_unit = pixel_unit
        
    def process_frame(self, frame):
        """Process a frame with YOLO detection and DeepSORT tracking"""
        # Run YOLO detection
        results = self.model.predict(frame, conf=self.confidence, classes=self.classes, verbose=False)[0]
        
        # Extract detections for DeepSORT
        detections = []
        for r in results.boxes.data.cpu().numpy():
            x1, y1, x2, y2, conf, cls = r
            detections.append(([x1, y1, x2-x1, y2-y1], conf, int(cls)))
        
        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=frame)
        
        # Prepare visualization
        frame_with_boxes = frame.copy()
        
        # Initialize BEV frame if calibration is set
        if self.bev_transformer is None and self.bev_coords is not None:
            self.bev_transformer = BirdsEyeView(frame, self.bev_coords, self.bev_size)
        
        bev_frame = None
        if self.bev_transformer is not None:
            bev_frame = self.bev_transformer.img2bird()
        
        # Draw bounding boxes and tracks
        for track in tracks:
            if not track.is_confirmed():
                continue
            
            track_id = track.track_id
            ltrb = track.to_ltrb()
            
            # Draw on original frame
            x1, y1, x2, y2 = map(int, ltrb)
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame_with_boxes, f"ID: {track_id}", (x1, y1-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Project bottom center to BEV if calibrated
            if bev_frame is not None:
                bottom_center = (int((x1 + x2) / 2), y2)
                try:
                    bev_point = self.bev_transformer.projection_on_bird(bottom_center)
                    cv2.circle(bev_frame, bev_point, 5, (0, 0, 255), -1)
                    cv2.putText(bev_frame, f"ID: {track_id}", 
                                (bev_point[0]+5, bev_point[1]-5), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                except:
                    pass
        
        return frame_with_boxes, bev_frame

# ## Main Processing Function

def process_video(video_path, output_path=None, yolo_model="yolov8n.pt", confidence=0.3, classes=None):
    """Process video with YOLO detection, DeepSORT tracking and BEV visualization"""
    
    # Extract background for calibration
    print("Extracting background for calibration...")
    background = calcBackground(video_path)
    
    # Get ROI
    print("Select ROI (Region of Interest):")
    roi, roi_coords = getROI(background)
    
    if roi is None:
        print("ROI selection failed. Using full frame.")
        roi = background
        h, w = background.shape[:2]
        roi_coords = [[0, 0], [w, h]]
    
    # Perform manual calibration
    print("Select 4 points that form a rectangle in the real world:")
    bev_coords, bev_size = manualCalibration(roi)
    
    # Initialize tracker
    tracker = ObjectTracker(yolo_model=yolo_model, confidence=confidence, classes=classes)
    tracker.set_bev_calibration(bev_coords, bev_size)
    
    # Process video
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Setup output video writer if specified
    out = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width*2, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Apply ROI if needed
        if roi_coords:
            x1, y1 = roi_coords[0]
            x2, y2 = roi_coords[1]
            frame_roi = frame[y1:y2, x1:x2]
        else:
            frame_roi = frame
        
        # Process frame
        result_frame, bev_frame = tracker.process_frame(frame_roi)
        
        # Resize BEV frame to match original height
        if bev_frame is not None:
            bev_frame = cv2.resize(bev_frame, (height, height))
            
            # Combine frames side by side
            combined_frame = np.zeros((height, width*2, 3), dtype=np.uint8)
            combined_frame[:result_frame.shape[0], :result_frame.shape[1]] = result_frame
            h_offset = (height - bev_frame.shape[0]) // 2
            w_offset = width + (width - bev_frame.shape[1]) // 2
            combined_frame[h_offset:h_offset+bev_frame.shape[0], 
                          w_offset:w_offset+bev_frame.shape[1]] = bev_frame
        else:
            combined_frame = np.hstack((result_frame, np.zeros_like(result_frame)))
        
        # Display result
        cv2.imshow("Tracking with BEV", combined_frame)
        
        # Write frame to output video
        if out:
            out.write(combined_frame)
        
        if cv2.waitKey(1) & 0xFF == 27:  # ESC key
            break
    
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

# ## Upload Files to Colab

from google.colab import files

# Upload video file
uploaded = files.upload()
video_path = next(iter(uploaded.keys()))

# Optional: Upload satellite image if using satellite-based calibration
use_satellite = input("Do you want to use satellite-based calibration? (y/n): ").lower() == 'y'
satellite_path = None
if use_satellite:
    uploaded_sat = files.upload()
    satellite_path = next(iter(uploaded_sat.keys()))

# ## Run the Main Processing Function

# Set YOLO model and confidence threshold
yolo_model = "yolov8n.pt"  # You can change to other models like yolov8m.pt, yolov8l.pt, etc.
confidence = 0.3

# Set classes to detect (0=person, 2=car, etc.)
# Leave as None to detect all classes
classes = [0, 2, 3, 5, 7]  # person, car, motorcycle, bus, truck

# Process video
output_path = "output_tracking.mp4"
process_video(video_path, output_path, yolo_model, confidence, classes)

# Download the processed video
files.download(output_path)
