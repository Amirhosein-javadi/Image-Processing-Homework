import  numpy as np
import  cv2
import scipy
from scipy import signal
import skimage.segmentation

def find_edge(matrix):
    edge_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    edge = np.abs(scipy.signal.convolve2d(matrix,edge_filter,mode='same'))
    edge = edge * 255 / np.max(edge)
    return edge

def find_best_centers(matrix,edge,K):
    [rows,cols,n] = matrix.shape
    number = int(K**0.5)
    s = max(rows,cols) // number
    tr = rows // (number*5)
    x = []
    y = []
    for i in range(number):
        for j in range(number): 
            x.append(i * cols // number + cols // (2*number) )
            y.append(j * rows // number + rows // (2*number) )
    x = np.uint16(x)
    y = np.uint16(y)
    for i in range(n**2):
            indx = np.where(edge[y[i]-tr:y[i]+tr , x[i]-tr:x[i]+tr] == np.min(edge[y[i]-tr:y[i]+tr , x[i]-tr:x[i]+tr]))
            indx = np.array(indx)
            y_new = indx[0,len(indx[0,:])//2]
            x_new = indx[1,len(indx[0,:])//2]
            x[i] = x[i]-tr+x_new
            y[i] = y[i]-tr+y_new
    return x,y,s

def energy_func(matrix,idx,x,y,s,k,alpha):
    Labels = np.zeros([rows,cols,2]) # label , energy
    Labels[:,:,1] = 10**50
    for i in range(k):
        xmin = max(x[i]-s,0)
        xmax = min(x[i]+s,cols) 
        ymin = max(y[i]-s,0)
        ymax = min(y[i]+s,rows)
        dlab = ((matrix[ymin:ymax,xmin:xmax,0]-matrix[y[i],x[i],0])**2 + (matrix[ymin:ymax,xmin:xmax,1]-matrix[y[i],x[i],1])**2 + (matrix[ymin:ymax,xmin:xmax,2]-matrix[y[i],x[i],2])**2) 
        dxy  = ((idx[0,ymin:ymax,xmin:xmax]-y[i])**2 + (idx[1,ymin:ymax,xmin:xmax]-x[i])**2) 
        D    = dlab + alpha * dxy 
        Condition = D[:ymax-ymin,:xmax-xmin] < Labels[ymin:ymax,xmin:xmax,1]
        Labels[ymin:ymax,xmin:xmax,1] = Condition  * D[:ymax-ymin,:xmax-xmin] +  ~Condition *  Labels[ymin:ymax,xmin:xmax,1]
        Condition = D[:ymax-ymin,:xmax-xmin] == Labels[ymin:ymax,xmin:xmax,1]
        Labels[ymin:ymax,xmin:xmax,0] = Condition * i + ~Condition *  Labels[ymin:ymax,xmin:xmax,0]
    smooth_label = scipy.signal.medfilt2d(Labels[:,:,0],25) 
    return smooth_label

def basic_func(k):
    n = int(k**0.5)
    x,y,s = find_best_centers(image,edge,n**2)
    alpha = 0.01
    idx      = np.indices((rows,cols)).reshape(2,rows,cols) 
    labels = energy_func(lab_image,idx,x,y,s,n**2,alpha)
    final_im = image.copy()
    boundaries = skimage.segmentation.find_boundaries(labels, mode='thick').astype(bool)
    final_im[:,:,0] = boundaries * 255 + ~boundaries * final_im[:,:,0]
    final_im[:,:,1] = boundaries * 255 + ~boundaries * final_im[:,:,1]
    final_im[:,:,2] = boundaries * 255 + ~boundaries * final_im[:,:,2]
    return final_im
    


image = cv2.imread('slic.jpg')
[rows,cols,n] = image.shape
lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
edge = find_edge(lab_image[:,:,0])
k = 64
im = basic_func(k)
cv2.imwrite('Result1.jpg',im)
k = 256
im = basic_func(k)
cv2.imwrite('Result2.jpg',im)
k = 1024
im = basic_func(k)
cv2.imwrite('Result3.jpg',im)
k = 2056
im = basic_func(k)
cv2.imwrite('Result4.jpg',im)