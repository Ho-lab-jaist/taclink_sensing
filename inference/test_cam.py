import cv2

cam = cv2.VideoCapture(0)

while cam.isOpened():    
    frame = cam.read()[1]
    cv2.imshow("Camera", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
	    
cam.release()
cv2.destroyAllWindows()
