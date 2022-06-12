import  numpy as np
import  cv2
import scipy
from scipy import signal
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import spsolve
import time
def stage1(matrix,laplacian,n):
    new_coordination = np.zeros_like(matrix).astype(np.uint32)
    [rows,cols] = matrix.shape
    gradient = np.zeros([n,3])
    counter = 1
    indx_target = np.zeros([2,n]).astype(np.uint32)
    indx_source = np.zeros([1,n]).astype(np.uint32)
    for i in range(rows):
        for j in range(cols):
            if matrix[i,j] == 1:
                new_coordination[i,j] =  counter
                gradient[counter-1] = laplacian[i,j,:]
                indx_target[0,counter-1] = i
                indx_target[1,counter-1] = j
                indx_source[0,counter-1] = counter-1
                counter = counter + 1
    return new_coordination,gradient,indx_target,indx_source

def stage2(matrix,gradient,coefficients,n_mask,mask):
    [rows,cols] = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i,j] == 1: 
                coefficients[n_mask[i,j]-1,n_mask[i,j]-1] = -4
                if mask[i+1,j] == 1:
                   coefficients[n_mask[i,j]-1,n_mask[i+1,j]-1] = 1
                else: 
                    gradient[n_mask[i,j]-1,:] = gradient[n_mask[i,j]-1,:] - target[i+1,j,:]
                if mask[i-1,j] == 1:
                    coefficients[n_mask[i,j]-1,n_mask[i-1,j]-1] = 1
                else: 
                    gradient[n_mask[i,j]-1,:] = gradient[n_mask[i,j]-1,:] - target[i-1,j,:] 
                if mask[i,j+1] == 1:
                    coefficients[n_mask[i,j]-1,n_mask[i,j]+1-1] = 1
                else: 
                    gradient[n_mask[i,j]-1,:] = gradient[n_mask[i,j]-1,:] - target[i,j+1,:]             
                if mask[i,j-1] == 1:
                    coefficients[n_mask[i,j]-1,n_mask[i,j]-2] = 1
                else: 
                    gradient[n_mask[i,j]-1,:] = gradient[n_mask[i,j]-1,:] - target[i,j-1,:]                
    return gradient,coefficients
                       
def stage3(matrix,coefficients,n_mask):                       
    [rows,cols] = matrix.shape
    for i in range(rows):
        for j in range(cols):
            if matrix[i,j] == 1: 
                coefficients[n_mask[i,j]-1,n_mask[i,j]-1] = -4
                coefficients[n_mask[i,j]-1,n_mask[i+1,j]-1] = 1
                coefficients[n_mask[i,j]-1,n_mask[i-1,j]-1] = 1
                coefficients[n_mask[i,j]-1,n_mask[i,j]-1+1] = 1
                coefficients[n_mask[i,j]-1,n_mask[i,j]-1-1] = 1
    return coefficients

def construct_image(source,target,mask):
    [s_row,s_cols,m]= np.shape(source)
    [t_row,t_cols,m]= np.shape(target)
    image = np.copy(target)
    image[t_row//2-s_row//2:t_row//2-s_row//2+s_row,t_cols//2-s_cols//2:t_cols//2-s_cols//2+s_cols] = mask*source + (1-mask)*target[t_row//2-s_row//2:t_row//2-s_row//2+s_row,t_cols//2-s_cols//2:t_cols//2-s_cols//2+s_cols]
    return image

start = time.time()              
source = cv2.imread('1.source.jpg').astype(np.int16)
mask = cv2.imread('mask.jpg')
target = cv2.imread('2.target.jpg')
# Quantization
mask = ((mask[:,:,0] > 255//2)).astype(np.uint8)

l_filter = np.array([[0,1,0],
                     [1,-4,1],
                     [0,1,0]])
laplacian = np.zeros_like(source)
laplacian[:,:,0] = scipy.signal.convolve2d(source[:,:,0],l_filter,mode='same')
laplacian[:,:,1] = scipy.signal.convolve2d(source[:,:,1],l_filter,mode='same')
laplacian[:,:,2] = scipy.signal.convolve2d(source[:,:,2],l_filter,mode='same')
# outlayer mask
kernel = np.array([[0,1,0],[1,1,1],[0,1,0]],dtype = np.uint8)
erosion = cv2.erode(mask,kernel,iterations = 1)
outlayer = mask - erosion
# coefficients
coefficients_size = np.sum(mask)
coefficients = np.zeros([coefficients_size,coefficients_size],dtype=np.int8)
# stage1
numbered_mask,gradient,indx_target,indx_source = stage1(mask,laplacian,coefficients_size)
# stage2
gradient,coefficients  = stage2(outlayer,gradient,coefficients,numbered_mask,mask)
# stage3
coefficients  = stage3(erosion,coefficients,numbered_mask)
# sparce coefficients and normalizing
coefficients = csc_matrix(coefficients)
b_gradian = csc_matrix(np.reshape(gradient[:,0],[coefficients_size,1]))
b_channel = spsolve(coefficients,b_gradian)
g_gradian = csc_matrix(np.reshape(gradient[:,1],[coefficients_size,1]))
g_channel = spsolve(coefficients,g_gradian)
r_gradian = csc_matrix(np.reshape(gradient[:,2],[coefficients_size,1]))
r_channel = spsolve(coefficients,r_gradian)
im_b  = source[:,:,0]
im_b[tuple(indx_target)] = b_channel[tuple(indx_source)]
im_b = cv2.normalize(im_b,0,255,norm_type=cv2.NORM_MINMAX)
im_g  = source[:,:,1]
im_g[tuple(indx_target)] = g_channel[tuple(indx_source)]
im_g = cv2.normalize(im_g,0,255,norm_type=cv2.NORM_MINMAX)
im_r  = source[:,:,2]
im_r[tuple(indx_target)] = r_channel[tuple(indx_source)]
im_r = cv2.normalize(im_r,0,255,norm_type=cv2.NORM_MINMAX)
new_source = np.zeros_like(source)
new_source[:,:,0] = im_b
new_source[:,:,1] = im_g
new_source[:,:,2] = im_r
mask_3d = np.zeros([255,1020,3])
mask_3d[:,:,0] = erosion
mask_3d[:,:,1] = erosion
mask_3d[:,:,2] = erosion
final_im = construct_image(new_source,target,mask_3d)
cv2.imwrite('Result.jpg',final_im)
end = time.time()
print(end - start)