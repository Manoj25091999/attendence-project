#!/usr/bin/env python
# coding: utf-8

# In[1]:


import face_recognition
import cv2
import numpy as np
import os
from datetime import datetime #for attendence


# In[ ]:


path = 'Imageattendence'
images = []
classnames = []
mylist = os.listdir(path)

for cl in mylist:
    curimg = cv2.imread(f'{path}/{cl}')
    images.append(curimg)
    classnames.append(os.path.splitext(cl)[0])

print(classnames)
print(images)


# In[ ]:


def imgencod(images):
    encodelist=[]
    for img in images:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encod = face_recognition.face_encodings(img)[0]
        encodelist.append(encod)
    return encodelist        
    


# In[ ]:



def markattendence(name):
    



encodelistknown = imgencod(images)
print('Encode Complete')


# ## Using the webcam to capture a new face for attendence

# In[ ]:


cap = cv2.VideoCapture(0)

while True:
    caps, frame = cap.read()
    imgS = cv2.resize(frame, (0,0), None, 0.25,0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
    
    faceincurframe = face_recognition.face_locations(imgS)
    encodscurframe = face_recognition.face_encodings(imgS,faceincurframe)
    
    for encodface, faceloc in zip(encodscurframe, faceincurframe):
        matches = face_recognition.compare_faces(encodelistknown,encodface)
        facedis = face_recognition.face_distance(encodelistknown,encodface)
        print(facedis)
        
        matchindex = np.argmin(facedis)
        
        # displaying the classname if a match is found
        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)
            # Building a rectangle around the face using face location
            y1,x2,y2,x1 = faceloc #Getting the coordinates
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,255,0),3)
            cv2.rectangle(frame,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            # Putting the text around the box using opencv
            cv2.putText(frame, name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,2,(0,255,0),3)
        
    # showing the matched image using opencv
    cv2.imshow('webcam',frame)
    cv2.waitKey(1)
        
    
    


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:


mylist


# In[ ]:


# Loading all the images
img_hrithik1 = face_recognition.load_image_file('imagedetec/Hrithik-Roshan-1.jpg') #bgr #returns an array representation of images
img_hrithik1 = cv2.cvtColor(img_hrithik1, cv2.COLOR_BGR2RGB) # BGR TO RGB
img_hrithik2 = face_recognition.load_image_file('imagedetec/Ratan_tata.jpg')
img_hrithik2 = cv2.cvtColor(img_hrithik2, cv2.COLOR_BGR2RGB)
img_ratantata = face_recognition.load_image_file('imagedetec/Ratan_tata.jpg')
img_ratantata = cv2.cvtColor(img_ratantata, cv2.COLOR_BGR2RGB)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




