from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("trained_model_on_spacecraft.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam


results = model.track(source="https://www.youtube.com/watch?v=R6K4M4Q09Kw", show=True)