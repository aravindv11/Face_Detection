import cv2

#load some pre-trained data on face frontals from opencv
trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
smile_Detector=cv2.CascadeClassifier('haarcascade_smile.xml')
eye_Detector=cv2.CascadeClassifier('haarcascade_eye.xml')


#capture video frame
webcam = cv2.VideoCapture(0)

#Iterate 
while True:
    # read current frame
    successful_frame_read ,frame = webcam.read()

    if not successful_frame_read:
        break

    #convert to Gray
    grayscaled_img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY) 

    #detect faces
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    
    #Draw rectangle
    for (x,y,w,h) in face_coordinates:
      cv2.rectangle(frame ,(x, y),(x+w, y+h),(0, 255, 0), 2)
      
      #get the sub-frame
      the_face=frame[y:y+h, x:x+w] 
      
      #gray scale
      face_img = cv2.cvtColor(the_face,cv2.COLOR_BGR2GRAY) 

      smile = smile_Detector.detectMultiScale(face_img, scaleFactor=1.7,minNeighbors= 20)
      
      eyes=eye_Detector.detectMultiScale(face_img, scaleFactor=1.1,minNeighbors= 10)

      #draws rectangle when smiling
      # for (x_,y_,w_,h_) in smile:
      #  cv2.rectangle(the_face ,(x_, y_),(x_+w_, y_+h_),(50, 50, 200), 2)
     
      #draws rectangle for eyes
      for (x_,y_,w_,h_) in eyes:
         cv2.rectangle(the_face ,(x_, y_),(x_+w_, y_+h_),(255, 255, 255), 2)
     
    #prints Smiling when detects smile
    if len(smile) > 0:
          cv2.putText(frame,'Smiling',(x,y+h+40),fontScale=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX,color=(0,0,0))
    
    #opens new window
    cv2.imshow('Face Detection ',frame)
    key=cv2.waitKey(1)
    
    #Stop if Q key is pressed
    if key==81 or key==113:
        break

# release webcam
webcam.release()   
cv2.destroyAllWindows()     
print("Code completed")