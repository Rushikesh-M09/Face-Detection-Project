import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


BaseOptions = python.BaseOptions
FaceDetector = vision.FaceDetector
FaceDetectorOptions = vision.FaceDetectorOptions
RunningMode = vision.RunningMode

model_path = "blaze_face_short_range.tflite"


face_detector = FaceDetector.create_from_options(
    FaceDetectorOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=RunningMode.VIDEO,
        )
)

#Replace "0" with video file path if needed
cap = cv2.VideoCapture(0)

fps = cap.get(cv2.CAP_PROP_FPS)
if fps == 0:
    fps = 30  # Default to 30 if FPS is not available

timestamp = 0

while True:
    ret, img = cap.read()
    if not ret:
        print("End of video or cannot read frame.")
        break
    
    key = cv2.waitKey(1)
    
    # Resize & convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
    
    # Run face detection
    results = face_detector.detect_for_video(mp_image, timestamp)
    
    # Draw detections
    if results.detections:
        for detection in results.detections:
            bbox = detection.bounding_box
            x1 = int(bbox.origin_x)
            y1 = int(bbox.origin_y)
            x2 = int(bbox.origin_x + bbox.width)
            y2 = int(bbox.origin_y + bbox.height)
            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 1)
            cv2.putText(img, "Face", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


    cv2.imshow("Face Detection", img)
    
    timestamp += int(1000 / fps)
    
    # Press 'q' to quit
    if key == ord('q'):
        break

cap.release()
face_detector.close()
cv2.destroyAllWindows()
print("processing completed...!")