import streamlit as st
from PIL import Image
import numpy as np
import cv2
from io import BytesIO

# Constants
DEMO_IMAGE = 'stand.jpg'
BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }
POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]
width, height = 368, 368

# Load pre-trained model with error handling
try:
    net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# App title and description
st.title("Human Pose Estimation with OpenCV")
st.text("Upload an image or use the demo image to estimate human poses.")

# Image uploader
img_file_buffer = st.file_uploader("Upload an image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])

# Load the image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    demo_image = DEMO_IMAGE
    image = np.array(Image.open(demo_image))

# Display the original image
st.subheader("Original Image")
st.image(image, caption="Original Image", use_column_width=True)

# Threshold slider
thres = st.slider("Confidence Threshold for Key Point Detection (0.0 - 1.0)", min_value=0, max_value=100, value=20, step=5) / 100

# Preprocess the image to fit the model
def preprocess_image(image, target_size):
    h, w = target_size
    scale = min(h / image.shape[0], w / image.shape[1])
    new_size = (int(image.shape[1] * scale), int(image.shape[0] * scale))
    resized_image = cv2.resize(image, new_size, interpolation=cv2.INTER_AREA)
    return resized_image

image = preprocess_image(image, (height, width))

# Pose detector function
@st.cache_data
def poseDetector(frame, threshold):
    frameWidth, frameHeight = frame.shape[1], frame.shape[0]
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (width, height), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom, partTo = pair
        idFrom, idTo = BODY_PARTS[partFrom], BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)

    t, _ = net.getPerfProfile()
    inference_time = t * 1000.0 / cv2.getTickFrequency()
    return frame, inference_time

# Run pose detection
output, inference_time = poseDetector(image, thres)

# Display the output image
st.subheader("Pose Estimation")
st.image(output, caption="Estimated Poses", use_column_width=True)

# Display inference time
st.text(f"Inference Time: {inference_time:.2f} ms")

# Add download button for processed image
output_image = Image.fromarray(output)
buf = BytesIO()
output_image.save(buf, format="JPEG")
byte_im = buf.getvalue()
st.download_button("Download Pose Estimation Image", byte_im, "pose_estimation.jpg", "image/jpeg")
