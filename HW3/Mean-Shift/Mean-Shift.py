import  numpy as np
import  cv2

image = cv2.imread('park.jpg')
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
image_b = cv2.resize(lab_image[:,:,0], (800,600), interpolation = cv2.INTER_AREA)
image_g = cv2.resize(lab_image[:,:,1], (800,600), interpolation = cv2.INTER_AREA)
image_r = cv2.resize(lab_image[:,:,2], (800,600), interpolation = cv2.INTER_AREA)

imm = np.zeros([600,800,3])
imm[:,:,0] = image_b 
imm[:,:,1] = image_g
imm[:,:,2] = image_r
imm = np.uint8(imm)
dst = imm.copy()
cv2.pyrMeanShiftFiltering(imm,20,12,dst)
dst = cv2.cvtColor(dst, cv2.COLOR_LAB2BGR)
result = np.zeros_like(image)
result[:,:,0] = cv2.resize(dst[:,:,0] , (4416,3312), interpolation = cv2.INTER_AREA)
result[:,:,1] = cv2.resize(dst[:,:,1] , (4416,3312), interpolation = cv2.INTER_AREA)
result[:,:,2] = cv2.resize(dst[:,:,2] , (4416,3312), interpolation = cv2.INTER_AREA)
cv2.imwrite('result.jpg',result)



