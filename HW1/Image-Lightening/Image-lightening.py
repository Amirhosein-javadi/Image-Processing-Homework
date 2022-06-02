import numpy as np
import cv2
pic = cv2.imread('Dark.jpg')
new_pic = np.uint8(np.floor(255 * np.log10(1+(pic)*0.085)/np.log10(1+255*0.085)))
cv2.imwrite('res01.jpg', new_pic) 