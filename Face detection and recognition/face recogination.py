import cv2
size = 4
webcam = cv2.VideoCapture(0) #Use camera 0

# We load the xml file
image = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))

    # detect MultiScale / faces
    faces =image.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup

        #cv2.rectangle(im, (x, y), (x + w, y + h),(0,255,0),thickness=4)
        cv2.rectangle(im, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)

        #Save just the rectangle faces in SubRecFaces
        sub_face = im[y:y+h, x:x+w]
        FaceFileName = "unknow_" + str(y) + ".jpg"
        #cv2.imwrite(FaceFileName, sub_face)

    # Show the image
    cv2.imshow('BCU Research by Waheed Rafiq (c)',   im)
    key = cv2.waitKey(10)
    # if Esc key is press then break out of the loop
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()