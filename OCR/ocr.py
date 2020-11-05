# -*- coding: utf-8 -*-
"""
Created on Wed Nov  4 15:40:33 2020

@author: kkoni
"""

import cv2 
import pytesseract
import matplotlib.pyplot as plt
from pytesseract import Output
 
img = cv2.imread('C:/Users/kkoni/Desktop/Python/OCR_hand.jpeg')
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

plt.imshow(img)
plt.show()

img2=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
plt.imshow(img2)
plt.show()



# =============================================================================
# for i in range(48):
#     print(f'Zakres od {10*i} do {10*i+10}')
#     color_from = (5*i,0,0)
#     color_to = (5*i+10,255,255)
#     mask = cv2.inRange(img, color_from, color_to)
#     result = cv2.bitwise_and(img, img, mask=mask)
#      
#     f = plt.figure(figsize=(15,5))
#     ax = f.add_subplot(121)
#     ax2 = f.add_subplot(122)
#     ax.imshow(mask, cmap="gray")
#     ax2.imshow(result)
#     plt.show()
# =============================================================================
    
# Adding custom options
custom_config = r'--oem 3 --psm 6'
pytesseract.pytesseract.tesseract_cmd = r'C:/Users/kkoni/AppData/Local/Tesseract-OCR/tesseract.exe'

d = pytesseract.image_to_data(img, output_type=Output.DICT)
print(d)
n_boxes = len(d['text'])
for i in range(n_boxes):
    if int(d['conf'][i]) > 20:
        (x, y, w, h) = (d['left'][i], d['top'][i], d['width'][i], d['height'][i])
        img = cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
        print(d['text'][i])
        
cv2.imshow('img', img)
cv2.waitKey(0)


