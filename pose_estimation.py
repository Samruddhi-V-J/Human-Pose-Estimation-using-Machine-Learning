import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the body parts and pose pairs
BODY_PARTS = {
    "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
    "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
    "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
    "LEye": 15, "REar": 16, "LEar": 17, "Background": 18
}

POSE_PAIRS = [
    ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
    ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
    ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
    ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
    ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"]
]

# Input dimensions for the model
inWidth = 368
inHeight = 368

# Load the pre-trained TensorFlow model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Confidence threshold for keypoint detection
thres = 0.2

def poseDetector(frame):
    """
    Detects human poses in the given image frame.

    Args:
        frame (numpy.ndarray): Input image frame.

    Returns:
        numpy.ndarray: Output frame with pose estimation overlay.
    """
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    # Pre-process the frame and forward it through the model
    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), 
                                       (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  # Extract relevant outputs
    
    assert len(BODY_PARTS) == out.shape[1]
    
    points = []
    for i in range(len(BODY_PARTS)):
        # Slice the heatmap of the corresponding body part
        heatMap = out[0, i, :, :]

        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > thres else None)
        
    # Draw skeleton by connecting key points
    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert partFrom in BODY_PARTS
        assert partTo in BODY_PARTS

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
    return frame

# Load input image
input_image_path = 'stand.jpg'  # Replace with your image path
input_image = cv2.imread(input_image_path)

if input_image is None:
    print("Error: Unable to load the input image. Check the path!")
    exit()

# Process the image
output_image = poseDetector(input_image)

# Save the output image
output_image_path = "OutPut-image.png"
cv2.imwrite(output_image_path, output_image)
print(f"Output saved to {output_image_path}")

# Display the output using OpenCV or Matplotlib
cv2.imshow("Pose Estimation", output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Optional: Display using Matplotlib
plt.imshow(cv2.cvtColor(output_image, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.title("Pose Estimation Result")
plt.show()
