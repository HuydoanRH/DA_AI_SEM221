import cv2
import os
import numpy as np
# import tensorflow as tf
from keras_preprocessing.image import img_to_array
from keras.models import load_model

os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"

def main():
	path = "./Test/test5.png"
	rawImage = cv2.imread(path, 1)

	model_FaceDetect = cv2.CascadeClassifier(cv2.data.haarcascades+ 'haarcascade_frontalface_default.xml')
	model_GenderRecog = load_model('gender_detect.model')
	face_image = detect_recog(rawImage, model_FaceDetect, model_GenderRecog)

	face_image = cv2.resize(face_image, (800,800))
	cv2.imshow("result", face_image)
	cv2.waitKey() 
	cv2.destroyAllWindows()
    
def detect_recog(image, model_detection, model_GenderRecog):
    classes = ['woman', 'man']
    img = image.copy()
    img_temp = image.copy ()
    img_temp = cv2.cvtColor(img_temp, cv2.COLOR_BGR2GRAY) # convert image to gray image

    #print(img_temp)
    face = model_detection.detectMultiScale(img_temp, scaleFactor=1.2, minNeighbors=2, minSize=(30, 30)) # face is a tupe if empty, else is a array

    # check have face or not
    if len(face) == 0:
        print("This picture doesn't contain any people")
        return img

    # get 4 corners face of each person
    for idx, person in enumerate(face):
        print("Detecting face %d:" % (idx + 1))
        startX = person[0]
        startY = person[1]
        endX = person[2]
        endY = person[3]
        print(person)
        print(person[1])

        # draw rectangle over face
        cv2.rectangle(img, (startX, startY), (startX + endX, startY + endY), (0, 255, 0), 2)

        # crop face 
        face_crop = img[startY:(startY + endY), startX:(startX + endX)]

        # resize 96x96 for input of recognition gender model
        print("shape[0]", face_crop.shape[0])
        print("shape[1]", face_crop.shape[1])
        print("shape[2]", face_crop.shape[2])
        if (face_crop.shape[0]) < 60 or (face_crop.shape[1]) < 60:
            continue

        face_crop = cv2.resize(face_crop, (96,96))

        # # show face image detected
        # cv2.imshow("crop_image", face_crop)
        # cv2.waitKey()
        # cv2.destroyAllWindows()

        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)

      # gender recognition
        gender = model_GenderRecog.predict(face_crop, batch_size = 32) # return a list
        print(gender)
        index = np.argmax(gender)
        print("predic value", gender)
        print(index)
        gender_class = classes[index]
        percent = gender[0][index]*100

        # label = conf[idx] * 100 + "," + label
        label = "{gender},{percent}".format(gender=gender_class, percent="{:.2f}%".format(percent))
        Y_pos = startY - 10 if startY - 10 > 10 else startY + 10

        # write label and confidence above face rectangle
        cv2.putText(img, label, (startX, Y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  

    return img

if __name__ == '__main__':
    main()
