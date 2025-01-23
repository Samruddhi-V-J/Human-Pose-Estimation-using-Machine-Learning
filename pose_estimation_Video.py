import cv2
import numpy as np

# Body parts and pose pairs
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
confidence_threshold = 0.2

# Load the pre-trained TensorFlow model
net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

def pose_estimation_video(cap):
    """
    Processes a video to detect and overlay pose estimations on each frame.

    Args:
        cap (cv2.VideoCapture): Video capture object.
    """
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("End of video or cannot read the video.")
            break

        # Resize frame for faster processing
        frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]

        # Pre-process the frame and forward it through the model
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight),
                                           (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # Slice relevant outputs

        assert len(BODY_PARTS) == out.shape[1]

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap for each body part
            heatMap = out[0, i, :, :]

            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            points.append((int(x), int(y)) if conf > confidence_threshold else None)

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
                cv2.ellipse(frame, points[idFrom], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (5, 5), 0, 0, 360, (0, 0, 255), cv2.FILLED)

        # Display frame
        cv2.imshow('Pose Estimation', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# Main execution
if __name__ == "__main__":
    video_path = "run.mov"  # Replace with the path to your video file
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video file. Check the path!")
    else:
        pose_estimation_video(cap)
