#Import necessary libraries
from flask import Flask, render_template, Response
import cv2
import cv2
import numpy as np
from keras.models import model_from_json
from Recommender import GetActualSong
import time

#Initialize the Flask app
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('app.html', emotion_list = None)

@app.route('/video')
def video_feed():
    return Response(EmotionDetection(), mimetype='multipart/x-mixed-replace; boundary=frame')

def EmotionDetection():
    
    emotion_dict = {0: "Angry", 1: "Disgust", 2: "Fear", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprise"}

    # load json and create model
    json_file = open('./model/emotion_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    emotion_model = model_from_json(loaded_model_json)

    # load weights into new model
    emotion_model.load_weights("./model/emotion_model.h5")
    cap = cv2.VideoCapture(0)

    while True:
        # Capture a frame from the video stream
        ret, frame = cap.read()
        if not ret:
            break
        else:
            # Send the frame to the webpage
            
            face_detector = cv2.CascadeClassifier('./haarcascades/haarcascade_frontalface_default.xml')
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # detect faces available on camera
            num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

            # take each face available on the camera and Preprocess it
            for (x, y, w, h) in num_faces:
                cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
                roi_gray_frame = gray_frame[y:y + h, x:x + w]
                cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

                # predict the emotions
                emotion_prediction = emotion_model.predict(cropped_img)
                maxindex = int(np.argmax(emotion_prediction))
                print(emotion_dict[maxindex])
                emotion_list.append(emotion_dict[maxindex])
                cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    

@app.route('/getsongs', methods=['POST'])
def get_Songs():
    maxCount = 0
    index = -1  # sentinels
    for i in range(len(emotion_list)): 
        count = 1
        for j in range(i+1, len(emotion_list)):
            if(emotion_list[i] == emotion_list[j]):
                count += 1
            if(count > maxCount): 
                maxCount = count
                index = i
        if (maxCount > len(emotion_list)//2):
            print(emotion_list[index])
    result = GetActualSong(emotion_list[index])
    time.sleep(7)
    return render_template('app.html',emotion_list = result)


if __name__ == "__main__":
    emotion_list = []
    app.run(debug=True)