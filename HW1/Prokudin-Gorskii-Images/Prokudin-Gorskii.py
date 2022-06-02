import numpy as np
import cv2

def slice_in_three(matrix):   
     rows = np.floor(np.size(matrix,axis=0)/3).astype(np.integer)
     b = matrix[0:rows,:,0].astype(np.float)
     g = matrix[rows:2*rows,:,0].astype(np.float)
     r = matrix[2*rows:3*rows,:,0].astype(np.float)
     return b,g,r

def pyramid(a,b,c):
    a_prime=cv2.pyrDown(a)
    b_prime=cv2.pyrDown(b)
    c_prime=cv2.pyrDown(c)
    return a_prime,b_prime,c_prime

def transition(b,g,r,n,x_b,y_b,x_r,y_r):#
    for i in range(n):
        b,g,r = pyramid(b,g,r)
    rows,cols = r.shape
    transition_matrix_b = np.float32([[1,0,x_b],[0,1,y_b]])
    transition_matrix_r = np.float32([[1,0,x_r],[0,1,y_r]])
    b = cv2.warpAffine(b,transition_matrix_b,(cols,rows)) 
    r = cv2.warpAffine(r,transition_matrix_r,(cols,rows))                              
    error_b=np.zeros([20,20])
    error_r=np.zeros([20,20])
    for i in range(-10,10):
        for j in range(-10,10):
            transition_matrix_b = np.float32([[1,0,i],[0,1,j]])
            transition_matrix_r = np.float32([[1,0,i],[0,1,j]])
            transitioned_b = cv2.warpAffine(b,transition_matrix_b,(cols,rows))
            transitioned_r = cv2.warpAffine(r,transition_matrix_r,(cols,rows))
            error_b[np.int(j+10),np.int(i+10)] = np.sum(np.abs(transitioned_b-g))
            error_r[np.int(j+10),np.int(i+10)] = np.sum(np.abs(transitioned_r-g))
    delta_y_b,delta_x_b = np.where(error_b==error_b.min())-np.array([10]) 
    delta_y_r,delta_x_r = np.where(error_r==error_r.min())-np.array([10])         
    return (y_b+delta_y_b),(x_b+delta_x_b),(y_r+delta_y_r),(x_r+delta_x_r)
    
def  create(b,g,r):
    n=np.int(np.ceil(np.log2(min(np.size(r,axis=0),np.size(r,axis=1)/20))))
    x_r=0
    y_r=0
    x_b=0
    y_b=0
    while(n!=-1):
        x_b*=2
        y_b*=2
        x_r*=2
        y_r*=2
        y_b,x_b,y_r,x_r=transition(b,g,r,n,x_b,y_b,x_r,y_r)
        n=n-1
        
    rows,cols = r.shape    
    transition_matrix_b = np.float32([[1,0,x_b],[0,1,y_b]])
    b= cv2.warpAffine(b,transition_matrix_b,(cols,rows))  
    transition_matrix_r = np.float32([[1,0,x_r],[0,1,y_r]])
    r= cv2.warpAffine(r,transition_matrix_r,(cols,rows))  
    return y_b,x_b,y_r,x_r,b,r
   
image = cv2.imread('melons.tif')
image = np.uint8(image)
b,g,r = slice_in_three(image)
y_b,x_b,y_r,x_r,New_b,New_r=create(b,g,r)

rows,cols = r.shape
im=np.zeros([rows,cols,3]).astype(np.uint8)
im[:,:,0]=np.int8(New_b)
im[:,:,1]=np.int8(g)
im[:,:,2]=np.int8(New_r)
cv2.imwrite('res04.jpg',im)

