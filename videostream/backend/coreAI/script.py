import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
from keras.preprocessing import image
#Load the saved model

#video = cv2.VideoCapture(0)
print('into here')

class Detect():
        def __init__(self):
                self.model = tf.keras.models.load_model('/Users/duylinh/Desktop/firedetection/videostream/backend/coreAI/modelfiredetect.h5')
                print('into __init__')
        def dectect(self, frame):
                #Convert the captured frame into RGB
                im = Image.fromarray(frame, 'RGB')
                #Resizing into 224x224 because we trained the model with this image size.
                im = im.resize((224,224))
                img_array = image.img_to_array(im)
                img_array = np.expand_dims(img_array, axis=0) / 255
                probabilities = self.model.predict(img_array)[0]
                print(probabilities)
                #Calling the predict method on model to predict 'fire' on the image
                prediction = np.argmax(probabilities)
                #if prediction is 0, which means there is fire in the frame.
                if prediction == 0:  
                        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                        print('frame: ',frame)
                        return frame
                else:
                        return frame

