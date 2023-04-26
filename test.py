import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QComboBox
import dlib
from scipy.spatial import Delaunay
import sys
import time
from PyQt5.QtCore import pyqtSignal, QObject
import os,re
import cv2
import numpy as np
# Define the available filters
filters = ['None', 'Cartoonize', 'Face swap', 'Aging effect', 'Face morphing', 'Face distortion', "Third Eye"]
print(0.1)

class Camera(QWidget):
    
    closed = pyqtSignal()
    def __init__(self):
        
        print(1)
        super().__init__()
        print(2)

        # Initialize the camera and face detection
        
        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')
        # Initialize the GUI elements
        self.label = QLabel(self)
        print(3)
        self.comboBox = QComboBox(self)
        self.comboBox.addItems(filters)
        self.comboBox.currentIndexChanged.connect(self.handle_filter_change)
        self.button = QPushButton('Capture', self)
        self.button.clicked.connect(self.capture_image)
        self.layout = QVBoxLayout(self)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.comboBox)
        self.layout.addWidget(self.button)

        # Start the timer to update the camera feed
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)
        self.setLayout(self.layout)
        self.filter=None
        self.but=False



    

    def new_name(self):
        # Search for files in the current directory with names like "output#.jpg"
        pattern = re.compile(r'output(\d+)\.jpg')
        max_number = 0
        for filename in os.listdir():
            match = pattern.match(filename)
            if match:
                number = int(match.group(1))
                max_number = max(max_number, number)

        # Build the new filename with the incremented number
        new_number = max_number + 1
        new_filename = f'output{new_number}.jpg'
        return new_filename

    def handle_filter_change(self, index):
        self.filter = self.comboBox.currentText()
        
    def capture_image(self):
        # Capture an image from the camera
        ret, frame = self.cap.read()

        # Display the captured image in the GUI
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = QImage(frame.data, frame.shape[1], frame.shape[0], QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image)
        self.label.setPixmap(pixmap)
        
        self.but=True


    print("init")
    def cartoonize(image):
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Apply bilateral filtering to the grayscale image to reduce noise
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)

            # Apply edge detection to the filtered image using the Canny algorithm
            edges = cv2.Canny(filtered, 30, 100)

            # Apply a dilation operation to thicken the edges
            kernel = np.ones((3,3), np.uint8)
            dilated = cv2.dilate(edges, kernel, iterations=1)

            # Create a black and white image by thresholding the dilated image
            _, thresh = cv2.threshold(dilated, 50, 255, cv2.THRESH_BINARY)

            # Create a 3-channel image by replicating the black and white image across all channels
            thresh = cv2.merge((thresh, thresh, thresh))

            # Apply bitwise AND to the original image and the thresholded image to create a cartoon-like effect
            cartoon = cv2.bitwise_and(image, thresh)

            return cartoon


    def face_swap(image):
            # Convert the image to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale image
            faces = Camera.self.face_cascade.detectMultiScale(gray, 1.3, 5)

            # If no faces are detected, return the original image
            if len(faces) == 0:
                return image

            # Select the first face detected
            x, y, w, h = faces[0]

            # Extract the face from the original image
            face1 = image[y:y+h, x:x+w]

            # Load a second image to use for the face swap
            image2 = cv2.imread('face2.jpg')

            # Resize the second image to the size of the extracted face
            image2 = cv2.resize(image2, (w, h))

            # Swap the faces by replacing the extracted face with the resized second image
            image[y:y+h, x:x+w] = image2

            return image
    def aging_effect(self,frame):
            # Create a copy of the original image
            output = frame.copy()

            # Convert the image to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Apply a bilateral filter to smooth the image
            smoothed = cv2.bilateralFilter(gray, 10, 75, 75)

            # Compute the edges of the smoothed image using the Canny edge detector
            edges = cv2.Canny(smoothed, 30, 100)

            # Threshold the edges to create a mask
            mask = cv2.threshold(edges, 250, 255, cv2.THRESH_BINARY_INV)[1]

            # Convert the mask to 3 channels to be able to use it with addWeighted
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            # Create a gray color for the mask
            gray_mask = np.zeros_like(frame)
            gray_mask[:] = (192, 192, 192)

            # Blend the gray mask and the original image using the mask
            output = cv2.addWeighted(output, 0.5, gray_mask, 0.5, 0)
            output = cv2.addWeighted(output, 0.5, mask, 0.5, 0)

            return output
    def face_morphing(image1, image2):
            # Load the landmark detectors for both images
            detector = dlib.get_frontal_face_detector()
            predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

            # Detect the faces in both images
            gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
            faces1 = detector(gray1)
            faces2 = detector(gray2)

            # Ensure that exactly one face is detected in each image
            if len(faces1) != 1 or len(faces2) != 1:
                raise ValueError("Exactly one face must be present in each image")

            # Get the facial landmarks for each image
            landmarks1 = predictor(gray1, faces1[0])
            landmarks2 = predictor(gray2, faces2[0])

            # Convert the landmarks to NumPy arrays
            landmarks1 = np.array([(p.x, p.y) for p in landmarks1.parts()])
            landmarks2 = np.array([(p.x, p.y) for p in landmarks2.parts()])

            # Calculate the average landmarks
            landmarks_avg = (landmarks1 + landmarks2) / 2

            # Calculate the Delaunay triangulation of the average landmarks
            rect = (0, 0, image1.shape[1], image1.shape[0])
            tri = Delaunay(landmarks_avg)

            # Initialize the output image as a grayscale image
            output = np.zeros_like(gray1)

            # Warp and blend the faces
            for i, tri_indices in enumerate(tri.simplices):
                # Get the landmarks for this triangle
                tri_landmarks1 = landmarks1[tri_indices]
                tri_landmarks2 = landmarks2[tri_indices]
                tri_landmarks_avg = landmarks_avg[tri_indices]

                # Calculate the affine transforms for each image
                transform1 = cv2.getAffineTransform(tri_landmarks1.astype(np.float32),
                                                    tri_landmarks_avg.astype(np.float32))
                transform2 = cv2.getAffineTransform(tri_landmarks2.astype(np.float32),
                                                    tri_landmarks_avg.astype(np.float32))

                # Warp the triangles from each image to the average landmarks
                warp1 = cv2.warpAffine(image1, transform1, rect[:2] + tri_landmarks_avg.shape[::-1])
                warp2 = cv2.warpAffine(image2, transform2, rect[:2] + tri_landmarks_avg.shape[::-1])

                # Blend the warped triangles
                mask = np.zeros_like(output)
                cv2.fillConvexPoly(mask, tri_landmarks_avg.astype(np.int32), 255, 16, 0)
                output = cv2.seamlessClone(warp2, output, mask, tri_landmarks_avg.astype(np.float32), cv2.NORMAL_CLONE)

            # Convert the output to a BGR image
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

            return output
    

    def face_distortion(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_detector = self.face_cascade
        eyes_detector = self.eye_cascade
        faces = face_detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x,y,w,h) in faces:
            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Find eyes
            eyes = eyes_detector.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5)

            for (ex,ey,ew,eh) in eyes:
                # Extract eye region
                eye = face[ey:ey+eh, ex:ex+ew]

                # Resize the eye
                resized_eye = cv2.resize(eye, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_LINEAR)

                # Replace eye region in original image
                if resized_eye.shape[0] <= face.shape[0]-ey and resized_eye.shape[1] <= face.shape[1]-ex:
                    face[ey:ey+resized_eye.shape[0], ex:ex+resized_eye.shape[1]] = resized_eye

            # Replace face region in original image
            frame[y:y+h, x:x+w] = face

        return frame
    def add_third_eye(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

        for (x, y, w, h) in faces:
            # Extract face region
            face = frame[y:y+h, x:x+w]

            # Find eyes
            eyes = self.eye_cascade.detectMultiScale(face, scaleFactor=1.1, minNeighbors=5)

            # Check if two eyes are found
            if len(eyes) < 2:
                continue

            # Extract left and right eye regions
            left_eye = None
            right_eye = None
            for (ex, ey, ew, eh) in eyes:
                if ex < w/2:
                    left_eye = face[ey:ey+eh, ex:ex+ew]
                else:
                    right_eye = face[ey:ey+eh, ex:ex+ew]

            # Check if both eyes are found
            if left_eye is None or right_eye is None:
                continue

            # Resize eye
            third_eye = cv2.resize(right_eye, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

            # Get forehead region
            forehead_y = max(0, int(y-h/7))
            forehead_h = int(h/3)
            forehead_x = max(0, int(x+w/4))
            forehead_w = int(w/2)
            forehead = frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w]

            # Add third eye to forehead
            eye_h, eye_w, _ = third_eye.shape
            eye_x = int((forehead_w - eye_w) / 2)
            eye_y = int((forehead_h - eye_h) / 1)
            forehead[eye_y:eye_y+eye_h, eye_x:eye_x+eye_w] = third_eye

            # Replace face region in original image
            frame[forehead_y:forehead_y+forehead_h, forehead_x:forehead_x+forehead_w] = forehead

        return frame









    

    def update_frame(self):
        global but
        # Read a frame from the camera
        ret, frame = self.cap.read()
        if ret:
            # Convert the frame to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces in the grayscale frame
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            output=frame
            # Apply the selected filter to the face
            
            if self.filter == None:
                output = frame
            elif self.filter == 'Cartoonize':
                edges = cv2.Canny(gray, 100, 200)
                output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            elif self.filter == 'Face swap':
                output = self.face_swap(frame)
            elif self.filter == 'Aging effect':
                output = self.aging_effect(frame)
            elif self.filter == 'Face morphing':
                output = self.face_morphing(frame)
            elif self.filter == 'Face distortion':
                
                output = self.face_distortion(frame)
            elif self.filter == 'Third Eye':
                output = self.add_third_eye(frame)
                
            

            # Convert the output to RGB format
            output = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
            
            # Draw the output on the frame
            for (x, y, w, h) in faces:
                cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            #self.face_distortion(output)
            # Convert the frame to QImage and display it on the GUI
            height, width, channels = output.shape
            bytesPerLine = channels * width
            qImg = QImage(output.data, width, height, bytesPerLine, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(qImg)
            self.label.setPixmap(pixmap)
        # Read a frame from the camera
        ret, frame = self.cap.read()
        if ret:
            # Apply the selected filter to the face
            
            if self.filter == 'Cartoonize':
                output = Camera.cartoonize(frame)
            elif self.filter == 'Face swap':
                output = Camera.face_swap(frame)
            elif self.filter == 'Aging effect':
                output = Camera.aging_effect(self,frame)
            elif self.filter == 'Face morphing':
                output = Camera.face_morphing(frame)
            elif self.filter == 'Third Eye':
                output = Camera.add_third_eye(self, frame)
                
            else:
                output = frame

            # Save the output image to the file system
            #cv2.imwrite(self.new_name(), output)

            # Exit the application
            if self.but==True:
                self.but=False
                cv2.imwrite(self.new_name(), output)
                

                
            else:
                pass
            

if __name__ == '__main__':
    
    app = QApplication(sys.argv)
    camera = Camera()
    camera.show()
    camera.closed.connect(app.quit)
    while app.exec_():
        if not camera.isVisible():
            break
    sys.exit()


