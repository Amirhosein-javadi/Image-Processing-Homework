import cv2
import numpy as np
from scipy import signal

tiger = np.uint8(cv2.imread('Img1.jpg'))
girl  = np.uint8(cv2.imread('Img2.jpg'))

rows,cols = (girl.shape[0],girl.shape[1])
tiger = cv2.resize(tiger, (cols,rows), interpolation = cv2.INTER_AREA)

h = np.eye(3) 
h[0][2]= int(((2639 + 2154)-(3315 + 2643))/2)
h[1][2]= int(((1574 + 1607)-(1475 + 1418))/2)
tiger = cv2.warpPerspective(tiger, h, (cols,rows))
cv2.imwrite('q4_03_near.jpg',tiger)
cv2.imwrite('q4_04_far.jpg',girl)

b_tiger,g_tiger,r_tiger = [tiger[:,:,0],tiger[:,:,1],tiger[:,:,2]]
b_tiger_fft = np.fft.fftshift(np.fft.fft2(b_tiger))
g_tiger_fft = np.fft.fftshift(np.fft.fft2(g_tiger))
r_tiger_fft = np.fft.fftshift(np.fft.fft2(r_tiger))
amplitude_r_tiger = np.abs(r_tiger_fft)
log_amplitude_tiger = np.uint8(np.log(amplitude_r_tiger))
cv2.imwrite('q4_05_dft_near.jpg',10 * log_amplitude_tiger)

b_girl,g_girl,r_girl = [girl[:,:,0],girl[:,:,1],girl[:,:,2]]
b_girl_fft = np.fft.fftshift(np.fft.fft2(b_girl))
g_girl_fft = np.fft.fftshift(np.fft.fft2(g_girl))
r_girl_fft = np.fft.fftshift(np.fft.fft2(r_girl))
amplitude_r_girl = np.abs(r_girl_fft)
log_amplitude_girl = np.uint8( np.log(amplitude_r_girl))
cv2.imwrite('q4_06_dft_far.jpg',10 * log_amplitude_girl)

x_gausian = signal.gaussian(cols, std=rows//60).reshape(1,-1)
y_gausian = signal.gaussian(rows, std=cols//60).reshape(-1,1)
gausian2d =  np.matmul(y_gausian, x_gausian)
cv2.imwrite('Q4_07_highpass_0.03.jpg',200*(1-gausian2d))
cv2.imwrite('Q4_08_lowpass_0.03.jpg',200*gausian2d)

cutoff_tiger     = np.ones([rows,cols])
cutoff_tiger[ rows//2-50 :rows//2+50 , cols//2-50 :cols//2+50] = 1/2
cutoff_tiger[ rows//2-30 :rows//2+30 , cols//2-30 :cols//2+30] = 0 
b_tiger_filtered = (b_tiger_fft - b_tiger_fft * gausian2d) * cutoff_tiger
g_tiger_filtered = (g_tiger_fft - g_tiger_fft * gausian2d) * cutoff_tiger
r_tiger_filtered = (r_tiger_fft - r_tiger_fft * gausian2d) * cutoff_tiger

cutoff_girl     = np.zeros([rows,cols])
cutoff_girl[ rows//2-50 :rows//2+50 , cols//2-50 :cols//2+50] = 1/2
cutoff_girl[ rows//2-30 :rows//2+30 , cols//2-30 :cols//2+30] = 1
b_girl_filtered =  b_girl_fft * gausian2d * cutoff_girl
g_girl_filtered =  g_girl_fft * gausian2d * cutoff_girl
r_girl_filtered =  r_girl_fft * gausian2d * cutoff_girl

cv2.imwrite('Q4_09_highpass_cutoff.jpg',200*cutoff_tiger)
cv2.imwrite('Q4_10_lowpass_cutoff.jpg',200*cutoff_girl)
cv2.imwrite('Q4_11_highpassed.jpg',200*(1-gausian2d)*cutoff_tiger)
cv2.imwrite('Q4_12_lowpassed.jpg',200* gausian2d   *cutoff_girl)
cv2.imwrite('Q4_13_hybrid_frequency.jpg',10 * np.log(np.abs(r_girl_filtered+r_tiger_filtered)))

im_b = np.abs(np.fft.ifft2(np.fft.ifftshift(b_tiger_filtered + b_girl_filtered)))
im_g = np.abs(np.fft.ifft2(np.fft.ifftshift(g_tiger_filtered + g_girl_filtered)))
im_r = np.abs(np.fft.ifft2(np.fft.ifftshift(r_tiger_filtered + r_girl_filtered)))
im   = np.zeros([rows,cols,3])
im[:,:,0] = np.uint8(im_b)
im[:,:,1] = np.uint8(im_g)
im[:,:,2] = np.uint8(im_r)
small_im = cv2.resize(im, (cols//25,rows//25), interpolation = cv2.INTER_AREA)

cv2.imwrite('Q_14_hybrid_near.jpg',im)
cv2.imwrite('Q_15_hybrid_far.jpg',small_im)


