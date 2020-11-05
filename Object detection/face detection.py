# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 11:05:34 2020

@author: kkoni
"""
import cv2 as cv

face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt2.xml')
smile_cascade = cv.CascadeClassifier('haarcascade_smile.xml')

cap = cv.VideoCapture(0)

while True:
    ret, img = cap.read()
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    
    for (x,y,w,h) in faces:
        img = cv.rectangle(img,(x,y),(x+w,y+h),(190,0,0),2)
        faceROI = gray[y:y+h,x:x+w]
        
        #eyes = eye_cascade.detectMultiScale(faceROI)
        #for (x,y,w,h) in eyes:
        #    img=cv.rectangle(img,(x,y),(x+w,y+h),(120,0,0),2)
    
    cv.imshow('img',img)
    
    if cv.waitKey(30) & 0xFF == ord('q'):
        break
    
cap.release()
cv.destroyAllWindows()