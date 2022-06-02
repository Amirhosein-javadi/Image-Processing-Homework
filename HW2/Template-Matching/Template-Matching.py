import  numpy as np
import  cv2
import scipy
from scipy import signal


def slice_in_three(matrix):   
     b = matrix[:,:,0].astype(np.float)
     g = matrix[:,:,1].astype(np.float)
     r = matrix[:,:,2].astype(np.float)
     return b,g,r

def HSV(matrix):
    [B,G,R]=slice_in_three(matrix)
    Cmax= (R>=G)*(R>B)*R + (G>R)*(G>=B)*G + (B>=R)*(B>G)*B + (B==G)*(B==R) * G
    Cmin= (R<=G)*(R<B)*R + (G<R)*(G<=B)*G + (B<=R)*(B<G)*B + (B==G)*(B==R) * G
    delta=Cmax-Cmin
    delta -= (delta==0) *1
    H = (Cmax==R)   * (((G-B)/delta)%6)*60 +\
        (Cmax!=R)   * (Cmax==G) * (((B-R)/delta)+2)*60 +\
        (Cmax!=R)   * (Cmax!=G) * (Cmax==B) * (((R-G)/delta)+4)*60
    delta += (delta==-1) *1
    Cmax += (Cmax ==0) *1
    S = (delta/Cmax) * 100
    return H,S

def find_edge(matrix):
    edge_filter = np.array([[0,1,0],[1,-4,1],[0,1,0]])
    edge = np.abs(scipy.signal.convolve2d(matrix,edge_filter,mode='same'))
    edge[0,:]=edge[-1,:]=edge[:,0]=edge[:,-1]=0
    edge = edge * 255 / np.max(edge)
    return edge

def Normalized_Cross_Correlation(g,f,trshold,tr0,xsize_big,ysize_big,xsize_small,ysize_small):
    h = np.zeros_like(f)
    for j in range(ysize_big):
        for i in range(xsize_big):
            if(trshold[j,i]) and (ysize_small <= j <= ysize_big-ysize_small ) and (xsize_small <= i <= xsize_big-xsize_small) :
                h[j,i]= np.sum((g-np.average(g)) * (f[j-ysize_small:j+ysize_small+1,i-xsize_small:i+xsize_small+1]-np.average(f[j-ysize_small:j+ysize_small+1,i-xsize_small:i+xsize_small+1]))) \
                   / ((np.sum((g-np.average(g))**2) * np.sum((f[j-ysize_small:j+ysize_small+1,i-xsize_small:i+xsize_small+1]-np.average(f[j-ysize_small:j+ysize_small+1,i-xsize_small:i+xsize_small+1]))**2))**0.5)
                   
    h = (h>tr0) * h
    return h

def draw_rectangle(im,h,tr):  
    color = (0, 0, 255) 
    thickness = 2   
    while(1):
        y,x = np.where(h == np.max(h))
        y = y[0]
        x = x[0]
        if h[y,x]<tr:
            break
        else:
            im = cv2.rectangle(im, (x-50,y-150), (x+50,y+100), color, thickness) 
            h[y-400:y+400,x-50:x+50]=0
    return im

patch = cv2.imread('patch.png')
ship  = cv2.imread('Greek_ship.jpg')

H_patch,S_patch = HSV(patch)
patch_rows,patch_cols = H_patch.shape
patch_rows  = patch_rows  // 2
patch_cols  = patch_cols  // 2
H_ship ,S_ship  = HSV(ship)
ship_rows,ship_cols = H_ship.shape
edge_S = find_edge(S_ship)
edge_H = find_edge(H_ship)
threshold =(edge_H>10) * (edge_S>10)
h1 = Normalized_Cross_Correlation(H_patch,H_ship,threshold,0.65,ship_cols,ship_rows,patch_cols,patch_rows)

x_ratio =0.5
y_ratio = 0.5
rows,cols = H_ship.shape
rows = int(rows*y_ratio)
cols = int(cols*x_ratio)
H_ship_2 = cv2.resize(H_ship, (cols,rows), interpolation = cv2.INTER_AREA)
S_ship_2 = cv2.resize(S_ship, (cols,rows), interpolation = cv2.INTER_AREA)
edge_H_2 = find_edge(H_ship_2)
edge_S_2 = find_edge(S_ship_2)
threshold2 =(edge_H_2>10) * (edge_S_2>10)
h2 = Normalized_Cross_Correlation(H_patch,H_ship_2,threshold2,0.57,cols,rows,patch_cols,patch_rows)
h2 = cv2.resize(h2, (int(cols/x_ratio),int(rows/y_ratio)), interpolation = cv2.INTER_AREA)

h =  h1 +  h2


im = ship.copy()
im = draw_rectangle(im,h,0.6)
cv2.imwrite('Result.jpg',im)