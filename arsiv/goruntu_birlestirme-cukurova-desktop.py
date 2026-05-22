import cv2
import numpy as np 
import os
import matplotlib.pyplot as plt
from natsort import natsorted   #dosyaları doğru sıralamak için eklendi


path = r"parcalar/aa_urgup"

liste = natsorted(os.listdir(path))
img_array=[]
for img in liste:
        
        genisleme=16
        #img_array.append(cv2.imread(os.path.join(path,img)))
        img_ary=cv2.imread(os.path.join(path,img))
        img_array.append(img_ary[genisleme:544-genisleme,genisleme:544-genisleme])                   
        
frame_adedi=20
        


# hor = img_array[0]
# ver = []
# for i in range(4*4-1):
#     print(i)    
#     hor = np.hstack((hor,img_array[i+1]))
#     if i%4==0:
#         ver.append(hor)
#         hor=img_array[i]
#     cv2.imshow("Yatay",hor)
    
# for i in range(len(ver)):
#     son = np.vstack((ver[i],ver[i+1]))

res=[]
ver = []
hor = []
   
for i in range(frame_adedi):
    
    for j in range(frame_adedi):
        
        res.append(img_array[frame_adedi*i+j])
        print(frame_adedi*i+j)
    

k=1
hor =  res[0]   
while k<frame_adedi*frame_adedi:
    
    if k%frame_adedi==0:
        ver.append(hor)
        hor=res[k]
        k+=1
        
        
    hor = np.hstack((hor,res[k]))
    
    k+=1
ver.append(hor)
    
    
    

    

  
k=1
son =  ver[0]   
while k<frame_adedi:
    son = np.vstack((son,ver[k]))
   
    k+=1

#cv2.imshow("son",son)

cv2.imwrite("birlestirilmis{}.jpg".format(frame_adedi),son)        
            


