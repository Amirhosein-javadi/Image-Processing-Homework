import  numpy as np
import  cv2
import scipy
from scipy import signal
import time

def generate_laplacian(matrix):
    g_filter = np.array([[1,4,6,4,1],
                         [4,16,24,16,4],
                         [6,24,36,24,6],
                         [4,16,24,16,4],
                         [1,4,6,4,1]])/256
    blured_im = np.zeros_like(matrix)
    blured_im[:,:,0] = scipy.signal.convolve2d(matrix[:,:,0],g_filter,mode='same')
    blured_im[:,:,1] = scipy.signal.convolve2d(matrix[:,:,1],g_filter,mode='same')
    blured_im[:,:,2] = scipy.signal.convolve2d(matrix[:,:,2],g_filter,mode='same')
    laplacian_im = np.int32(matrix)-np.int32(blured_im)
    [rows,cols,m]= np.shape(matrix)
    resized_im = cv2.pyrDown(matrix)
    return blured_im,laplacian_im,resized_im

def construct_image(source,target,mask):
    [s_row,s_cols,m]= np.shape(source)
    [t_row,t_cols,m]= np.shape(target)
    image = np.copy(target)
    image[t_row//2-s_row//2:t_row//2-s_row//2+s_row,t_cols//2-s_cols//2:t_cols//2-s_cols//2+s_cols] = mask*source + (1-mask)*target[t_row//2-s_row//2:t_row//2-s_row//2+s_row,t_cols//2-s_cols//2:t_cols//2-s_cols//2+s_cols]
    return image

# source = cv2.imread('plane.png')
start = time.time() 
source1 = cv2.imread('1.source.jpg')
mask1 = cv2.imread('mask.jpg').astype(float)
target1 = cv2.imread('2.target.jpg')



#  mask bank
mask1 = ((mask1 > 255//2)).astype(float)
mask2 = ((cv2.pyrDown(mask1)>0.9)*1).astype(float)
mask3 = ((cv2.pyrDown(mask2)>0.8)*1).astype(float)
mask4 = ((cv2.pyrDown(mask3)>0.7)*1).astype(float)
mask5 = ((cv2.pyrDown(mask4)>0.6)*1).astype(float)
mask6 = ((cv2.pyrDown(mask5)>0.5)*1).astype(float)
#  source bank
blured_source1,laplacian_source1,source2 = generate_laplacian(source1)
blured_source2,laplacian_source2,source3 = generate_laplacian(source2)
blured_source3,laplacian_source3,source4 = generate_laplacian(source3)
blured_source4,laplacian_source4,source5 = generate_laplacian(source4)
blured_source5,laplacian_source5,source6 = generate_laplacian(source5)
#  target bank
blured_target1,laplacian_target1,target2 = generate_laplacian(target1)
blured_target2,laplacian_target2,target3 = generate_laplacian(target2)
blured_target3,laplacian_target3,target4 = generate_laplacian(target3)
blured_target4,laplacian_target4,target5 = generate_laplacian(target4)
blured_target5,laplacian_target5,target6 = generate_laplacian(target5)
#  filtered mask band
g = 9
filtered_mask1 = cv2.GaussianBlur(mask1,(g,g),0)
filtered_mask2 = cv2.GaussianBlur(mask2,(g,g),0)
filtered_mask3 = cv2.GaussianBlur(mask3,(g,g),0)
filtered_mask4 = cv2.GaussianBlur(mask4,(g,g),0)
filtered_mask5 = cv2.GaussianBlur(mask5,(g,g),0)
filtered_mask6 = cv2.GaussianBlur(mask6,(g,g),0)


final_im = construct_image(source6,target6,filtered_mask6).astype(float)
final_im = cv2.pyrUp(final_im)[0:-1]
laplacian = construct_image(laplacian_source5,laplacian_target5,filtered_mask5)
final_im = final_im + laplacian
final_im = cv2.pyrUp(final_im)[0:-1,0:-1]
laplacian = construct_image(laplacian_source4,laplacian_target4,filtered_mask4)
final_im = final_im + laplacian
final_im = cv2.pyrUp(final_im)[0:-1]
laplacian = construct_image(laplacian_source3,laplacian_target3,filtered_mask3)
final_im = final_im + laplacian
final_im = cv2.pyrUp(final_im)[0:-1]
laplacian = construct_image(laplacian_source2,laplacian_target2,filtered_mask2)
final_im = final_im + laplacian
final_im = cv2.pyrUp(final_im)
laplacian = construct_image(laplacian_source1,laplacian_target1,filtered_mask1)
final_im = final_im + laplacian
cv2.imwrite('Result.jpg',final_im)
end = time.time()
print(end - start)