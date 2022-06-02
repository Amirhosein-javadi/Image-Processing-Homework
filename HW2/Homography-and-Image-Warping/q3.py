import cv2
import numpy as np 

def find_index(row,cols,h):
    im_dst  = np.zeros([row,cols,3])
    idx_src     = np.indices(im_dst[:,:,0].shape).reshape(2, -1)
    idx_dst = np.append(idx_src, np.ones([1,np.size(idx_src,axis=1)]), axis=0)
    idx_dst = np.matmul(h, idx_dst)
    idx_dst[0:2,:] = (idx_dst[0:2,:]/idx_dst[2,:])
    idx_dst = np.delete(idx_dst, 2, 0)  
    return idx_src,idx_dst


def transform(row,cols,A,idx_src,idx_dst):
    A_b   = A[:,:,0]
    A_g   = A[:,:,1]
    A_r   = A[:,:,2]
    im_b  = np.zeros([row,cols])
    im_b[tuple(idx_src)] = A_b[tuple(idx_dst)]
    im_g  = np.zeros([row,cols])
    im_g[tuple(idx_src)] = A_g[tuple(idx_dst)]
    im_r  = np.zeros([row,cols])
    im_r[tuple(idx_src)] = A_r[tuple(idx_dst)]
    im    = np.zeros([row,cols,3])
    im[:,:,0]=im_b
    im[:,:,1]=im_g
    im[:,:,2]=im_r
    return im
    
Books = cv2.imread('books.jpg')
pts_src1 = np.array([[209,665], [394,601], [104,383],[288,316]])
pts_dst1 = np.array([[0, 0], [0,195], [300,0],[300,195]])
h1,status = cv2.findHomography(pts_dst1,pts_src1)
idx_src1,idx_dst1 = np.int16(find_index(300,195,h1))
im1 = transform(300,195,Books,idx_src1,idx_dst1)

im_dst2  = np.zeros([283,205,3])
pts_src2 = np.array([[741,357], [709,152], [464,404],[427,204]])
pts_dst2 = np.array([[0, 0], [0,205], [283,0],[283,205]])
h2,status = cv2.findHomography(pts_dst2,pts_src2)
idx_src2,idx_dst2 = np.int16(find_index(283,205,h2))
im2 = transform(283,205,Books,idx_src2,idx_dst2)

im_dst3  = np.zeros([357,240,3])
pts_src3 = np.array([[967,813], [1100,611], [667,623],[795,420]])
pts_dst3 = np.array([[0, 0], [0,240], [357,0],[357,240]])
h3,status = cv2.findHomography(pts_dst3,pts_src3)
idx_src3,idx_dst3 = np.int16(find_index(357,240,h3))
im3 = transform(357,240,Books,idx_src3,idx_dst3)

cv2.imwrite('res04.jpg',im1)
cv2.imwrite('res05.jpg',im2)
cv2.imwrite('res06.jpg',im3)





