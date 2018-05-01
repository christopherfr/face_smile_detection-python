# -*- coding: utf-8 -*-
"""
Created on Wed Jan 24 15:52:59 2018

@author: chfer
"""

import cv2

if __name__ == '__main__':
    # Face and smile models (THESE ARE NOT MINE)
    faceCascade = cv2.CascadeClassifier('models/haarcascade_frontalface_default.xml')
    smileCascade = cv2.CascadeClassifier('models/haarcascade_smile.xml')
    
    # Stream parameters
    ip = '192.168.1.10' # Your phone's IP might be different
    port = '8080' # "IP Webcam" uses this port by default
    address = 'http://' + ip + ':' + port + '/video'
    
    stream = cv2.VideoCapture(0)
    stream.open(address)
    
    # Check if camera opened successfully
    if (stream.isOpened() == False): 
        print("Error: Stream could not be opened")
    else:
        # Read and display video frames until stream ends or user quits by pressing ESC
        cv2.startWindowThread()
        while(stream.isOpened()):
            # Capture frame-by-frame
            retrieve, frame = stream.read()
            
            if retrieve:
                frameGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Face detection
                faces = faceCascade.detectMultiScale(frameGray, 1.2, 7)
                for (x, y, w, h) in faces:
                    # Draw rectangle on detected face 
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    
                    # Extract face so smile detection will only be performed inside of it
                    faceGray = frameGray[y: y + h, x: x + w]
                    # Smile detection
                    smile = smileCascade.detectMultiScale(faceGray, 1.2, 40)
                    for (xx, yy, ww, hh) in smile:
                        # Draw rectangle on detected smile
                        cv2.rectangle(frame, (x+xx, y+yy), (x+xx + ww, y+yy + hh), (0, 255, 0), 2)
                        
                # Display the resulting frame
                cv2.imshow('Android stream - face and smile detection',frame)
                
                if (cv2.waitKey(1) & 0xFF == 27):
                    break
            else:
                print("Error: Frame could not be retrieved")
                break
        
        stream.release()
        cv2.destroyAllWindows()