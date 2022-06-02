import numpy as np
import cv2
import matplotlib.pyplot as plt
Dark=cv2.imread('Dark.jpg')
Pink=cv2.imread('Pink.jpg')

D_Hist_b = np.cumsum(np.histogram(Dark[:,:,0],256,[0,256])[0])
D_Hist_g = np.cumsum(np.histogram(Dark[:,:,1],256,[0,256])[0])
D_Hist_r = np.cumsum(np.histogram(Dark[:,:,2],256,[0,256])[0])
   
     
P_Hist_b = np.floor(np.cumsum(np.histogram(Pink[:,:,0],256,[0,256])[0])* (np.size(Dark)/np.size(Pink)))
P_Hist_g = np.floor(np.cumsum(np.histogram(Pink[:,:,1],256,[0,256])[0])* (np.size(Dark)/np.size(Pink)))
P_Hist_r = np.floor(np.cumsum(np.histogram(Pink[:,:,2],256,[0,256])[0])* (np.size(Dark)/np.size(Pink)))


delta_b = np.zeros([256])
delta_g = np.zeros([256])
delta_r = np.zeros([256])
new_pic = np.copy(Dark)

for i in range(255):
    delta_b[i] = (np.abs(P_Hist_b-D_Hist_b[i])).argmin()    
    delta_g[i] = (np.abs(P_Hist_g-D_Hist_g[i])).argmin() 
    delta_r[i] = (np.abs(P_Hist_r-D_Hist_r[i])).argmin() 
    new_pic[:,:,0] += (Dark[:,:,0]==i) * (delta_b[i]-i).astype(np.uint8)
    new_pic[:,:,1] += (Dark[:,:,1]==i) * (delta_g[i]-i).astype(np.uint8)
    new_pic[:,:,2] += (Dark[:,:,2]==i) * (delta_r[i]-i).astype(np.uint8)

new_pic_b_Hist = np.histogram(new_pic[:,:,0],256,[0,256])[0]
new_pic_g_Hist = np.histogram(new_pic[:,:,1],256,[0,256])[0]
new_pic_r_Hist = np.histogram(new_pic[:,:,2],256,[0,256])[0]

             
cv2.imwrite('equalized.jpg',new_pic) 
contrast=np.arange(0, 256)
plt.plot(contrast, new_pic_b_Hist, 'b', contrast, new_pic_g_Hist, 'g', contrast, new_pic_r_Hist , 'r')
plt.savefig('res05.jpg')

