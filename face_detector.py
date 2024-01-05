from flask import Flask, request
from PIL import Image
import numpy as np
import io
import dlib
import cv2

app = Flask(__name__)
        
        
def isCentered(center_x, center_y, face_w, face_h, img_w, img_h):
    return abs(center_x - img_w // 2) < (face_w // 4) \
        and abs(center_y - img_h // 2) < (face_h // 4)
        
def isScaled(face_w, face_h, img_w, img_h):
    ratio_w = face_w / img_w
    ratio_h = face_h / img_h
    # print(ratio_h, ratio_w)
    return ratio_w > 0.23 and ratio_w < 0.4 and \
        ratio_h > 0.3 and ratio_h < 0.5

def isHorizontal(landmarks):
    eye_y = [landmarks.part(i).y for i in range(4)]
    eye_w = (abs(landmarks.part(0).x - landmarks.part(1).x) + abs(landmarks.part(2).x - landmarks.part(3).x)) / 2
    # print(max(eye_y) - min(eye_y), eye_w)
    return (max(eye_y) - min(eye_y)) < eye_w // 2
        
        
class Detector:
    def __init__(self) -> None:
        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor('./weights/shape_predictor_5_face_landmarks.dat')
               
    def detect(self, image):
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            face = self.detector(gray)[0]
            landmarks = self.predictor(gray, face) 
                       
            x, y = face.left(), face.top()
            face_w, face_h = face.width(), face.height()
            img_w, img_h = image.shape[1], image.shape[0]
            
            center_x = x + face_w // 2
            center_y = y + face_h // 2
            
            centered = isCentered(center_x, center_y, face_w, face_h, img_w, img_h)
            scaled = isScaled(face_w, face_h, img_w, img_h)
            horizontal = isHorizontal(landmarks)            
            passed = centered and scaled and horizontal
            
            info = []
            if not passed:
                if not centered:
                    info.append("The face is not in the center")
                if not scaled:
                    info.append("The face is too far/close to the camera")
                if not horizontal:
                    info.append("The face is not horizontal")
            else:
                info.append("The face is at the right position")
            
            return {"result": passed, "info": info}
        except:
            return {"result": False, "info": ["Fail to detect"]}
            
        


mDetector = Detector()
# image = cv2.imread('./input.png')
# print(mDetector.detect(image))

@app.route('/detect_face', methods=['POST'])

def detect_face():    
    data = request.json
    image_data = data['camera_frame']

    image = Image.open(io.BytesIO(base64.b64decode(image_data)))
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
    return mDetector.detect(cv_image)
    
    
if __name__ == '__main__':
    app.run()