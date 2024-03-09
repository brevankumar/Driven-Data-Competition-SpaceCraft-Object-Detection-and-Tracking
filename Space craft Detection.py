from ultralytics import YOLO
from PIL import Image
import cv2

model = YOLO("trained_model_on_spacecraft.pt")
# accepts all formats - image/dir/Path/URL/video/PIL/ndarray. 0 for webcam

im1 = Image.open("data/test/images/07637f713f619308a1c153c7bad5fafd.png")

im2 = Image.open("data/test/images/06868d9273817e9b47d8766505e2874a.png")


results = model.predict(source=[im1,im2], show=True,save=True, conf=0.5) # Display preds. Accepts all YOLO predict arguments



