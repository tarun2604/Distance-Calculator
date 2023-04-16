import cv2
import numpy as np


obj_height = 25  
obj_width = 15  


focal_length = 1000


cap = cv2.VideoCapture(0)

while True:
    
    ret, frame = cap.read()
    orgy = cv2.cvtColor(frame, cv2.COLORMAP_AUTUMN)
    

    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
    _, thresh = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)

    
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    
    if len(contours) > 0:
        c = max(contours, key=cv2.contourArea)

        
        area = cv2.contourArea(c)
        perimeter = cv2.arcLength(c, True)

    
    edges = cv2.Canny(gray, 50, 150)

    
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        
        pixel_height = h
        pixel_width = w

        
        distance = (obj_width * focal_length) / pixel_width

        
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(frame, "{:.2f} cm".format(distance), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    
    cv2.imshow('Distance Calculation', frame)

    
    cv2.imshow('Original Vision', orgy)
    

    
    key = cv2.waitKey(1)

    
    if key == ord('q'):
        break


cap.release()
cv2.destroyAllWindows()
 