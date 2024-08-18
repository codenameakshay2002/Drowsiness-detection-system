from flask import Flask, jsonify
from flask_cors import CORS
from scipy.spatial import distance
from imutils import face_utils
from pygame import mixer
import imutils
import dlib
import cv2

app = Flask(__name__)
CORS(app)  # Enable CORS

# Load music file
mixer.init()
mixer.music.load("D:/drowsiness/Driver Drowsiness Detection-20240308T072529Z-001/Driver Drowsiness Detection/music.wav")

# Define eye aspect ratio function
def eye_aspect_ratio(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

# Define mouth aspect ratio function
def mouth_aspect_ratio(mouth):
    A = distance.euclidean(mouth[2], mouth[10])  # 51, 59
    B = distance.euclidean(mouth[4], mouth[8])  # 53, 57
    C = distance.euclidean(mouth[0], mouth[6])  # 49, 55
    mar = (A + B) / (2.0 * C)
    return mar

# Initialize dlib's face detector and shape predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("D:/drowsiness/Driver Drowsiness Detection-20240308T072529Z-001/Driver Drowsiness Detection/models/shape_predictor_68_face_landmarks.dat")

# Define facial landmarks for left and right eyes
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Route for starting drowsiness detection
@app.route('/start-detection', methods=['POST'])
def start_detection():
    thresh = 0.25  # Define threshold here
    frame_check = 20  # Define frame check here
    
    cap = cv2.VideoCapture(0)
    flag = 0
    while True:
        ret, frame = cap.read()
        if not ret or frame is None:  # Check if frame is None or reading failed
            continue  # Skip this iteration and try again
        
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = detector(gray, 0)
        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            for (x, y) in shape:
                cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
            
            # Detect left eye
            leftEye = shape[lStart:lEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            
            # Detect right eye
            rightEye = shape[rStart:rEnd]
            rightEAR = eye_aspect_ratio(rightEye)
            
            ear = (leftEAR + rightEAR) / 2.0
            
            # Detect mouth
            mouth = shape[48:68]
            mouthMAR = mouth_aspect_ratio(mouth)
            
            # Display eye status text overlay
            if ear < thresh:
                cv2.putText(frame, "EYES CLOSED", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "EYES OPEN", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            if ear < thresh:
                flag += 1
                if flag >= frame_check:
                    mixer.music.play()  # Play alert sound
                    cv2.putText(frame, "****************ALERT!****************", (10, 60),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                flag = 0
                
            # Check for yawn
            MOUTH_AR_THRESH = 0.79
            if mouthMAR > MOUTH_AR_THRESH:
                cv2.putText(frame, "YAWNING!", (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
        
        # Display the frame
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)  # Create a resizable window
        cv2.setWindowProperty("Frame", cv2.WND_PROP_TOPMOST, 1)  # Set the window to be always on top
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return jsonify({'status': 'stopped'})

if __name__ == '__main__':
    app.run(debug=True)
