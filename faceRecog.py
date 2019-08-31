import cv2
import os
import numpy as np

subjects = ["","Bhavik","Avinash"]

def draw_rectangle(img, rect):
    (x,y,w,h) = rect
    cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0),2)
    
def draw_text(img, text, x, y):
    cv2.putText(img, text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5,(0,255,0),2)

def detect_face(img, image_path):
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray_color = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)   #lbpcascade_frontalface.xml
    face_cascade = cv2.CascadeClassifier('opencv-files\lbpcascade_frontalface.xml')
    faces = face_cascade.detectMultiScale(gray,scaleFactor=1.015, minNeighbors=5)
    
    for f in range(0, len(faces)): 
        (x,y,w,h) = faces[f]
        draw_rectangle(gray_color, faces[f])
        draw_text(gray_color, str(f+1),x,y )
    cv2.imshow("select face", gray_color)
    cv2.waitKey(1000)
    selection = int(input("Select the Face:"))
    if(selection==0):
        #delete the image
        try:
            os.remove(image_path)
        except:
            pass
    cv2.destroyAllWindows()
        
    
    
    if(len(faces)==0):
        print("No face")
        cv2.imwrite(str(gray.shape[0])+".jpg", img)
        return None, None
    
    (x,y,w,h)=faces[selection-1]
    return gray[y:y+w,x:x+h],faces[selection-1]

def prepare_training_data(data_folder_path):
    dirs = os.listdir(data_folder_path)
    faces = []
    labels = [] 
    
    for dir_name in dirs:
       # print(dir_name)
        if not dir_name.startswith("s"):
            continue
        label = int(dir_name.replace("s",""))
        subject_dir_path = data_folder_path + "/"+ dir_name
        subject_images_names = os.listdir(subject_dir_path)
       # print(subject_dir_path,subject_images_names)
        for image_name in subject_images_names:
            #print(image_name)
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path+"/"+image_name
            image = cv2.imread(image_path)
            
            cv2.imshow("Training on image...", image)
            cv2.waitKey(100)
            face,rect = detect_face(image,image_path)
            
            if face is not None:
                
                faces.append(face)
                labels.append(label)
                
    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()
    return faces, labels

print("Preparing Data..")
faces, labels = prepare_training_data("C:\\Users\\Akil Stark\\OneDrive\\Desktop\\jupyter\\training-data")
print("Data Prepared")

print("Total faces: ", len(faces))
print("Total labels: ", len(labels))

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces,np.array(labels))



    
def predict(test_img, image_path):
    img = np.copy(test_img)
    face, rect = detect_face(img,image_path)
    
    label = face_recognizer.predict(face)
    #print(label)
    label_text = subjects[label[0]]
    
    draw_rectangle(img, rect)
    draw_text(img, label_text, rect[0],rect[1]-5)
    return img 
print("Predicting Images..")

test_folder = "test-data"
imgs = os.listdir(test_folder)
for img in imgs:
    img_path = test_folder+"\\"+img
    test_img = cv2.imread(img_path)
    # test_img2 = cv2.imread("test-data\\test2.jpg")



    predicted_img1 = predict(test_img,img_path)
    #predicted_img2 = predict(test_img2,"test-data\\test2.jpg")

    print("Prediction Complete!!")

    cv2.imshow("prediction",predicted_img1)

    cv2.waitKey(0)
    cv2.destroyAllWindows()