ورimport  numpy as np
import  cv2
import time
import skimage.segmentation

def Fill_Image_Generate_Template(New_pic,y,x,flag):
    if flag == 'first texture':
        final_img[0:blocksize,0:blocksize] = New_pic
        template = final_img[0:blocksize,blocksize-overlap:blocksize] 
        return template
    
    
    if flag == 'top marginal texture':  
        boundaries = skimage.segmentation.find_boundaries(best_cut, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        blur = cv2.GaussianBlur(New_pic,(15,15),0)
        x = x*(blocksize-overlap)
        condition1 = best_cut * ~boundaries
        condition2 = ~best_cut * ~boundaries
        condition3 = boundaries
        final_img[0:blocksize,x:x+blocksize,0] = condition1 * New_pic[:,:,0] + condition2 * final_img[0:blocksize,x:x+blocksize,0] + condition3 * blur[:,:,0]
        final_img[0:blocksize,x:x+blocksize,1] = condition1 * New_pic[:,:,1] + condition2 * final_img[0:blocksize,x:x+blocksize,1] + condition3 * blur[:,:,1]
        final_img[0:blocksize,x:x+blocksize,2] = condition1 * New_pic[:,:,2] + condition2 * final_img[0:blocksize,x:x+blocksize,2] + condition3 * blur[:,:,2]
        template = final_img[0:blocksize,x+blocksize-overlap:x+blocksize]
        x = x_loc + 1
        return template,x
    
    
    if flag == 'left marginal texture': 
        boundaries = skimage.segmentation.find_boundaries(best_cut, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        blur = cv2.GaussianBlur(New_pic,(15,15),0)
        y = y*(blocksize-overlap)
        condition1 = best_cut * ~boundaries
        condition2 = ~best_cut * ~boundaries
        condition3 = boundaries
        final_img[y:y+blocksize,0:blocksize,0] = condition1 * New_pic[:,:,0] + condition2 * final_img[0:blocksize,x:x+blocksize,0] + condition3 * blur[:,:,0]
        final_img[y:y+blocksize,0:blocksize,1] = condition1 * New_pic[:,:,1] + condition2 * final_img[0:blocksize,x:x+blocksize,1] + condition3 * blur[:,:,1]
        final_img[y:y+blocksize,0:blocksize,2] = condition1 * New_pic[:,:,2] + condition2 * final_img[0:blocksize,x:x+blocksize,2] + condition3 * blur[:,:,2]
        template = final_img[y+blocksize-overlap:y+blocksize,0:blocksize]
        y = y_loc + 1
        return template,y
    
    
    if flag == 'middle marginal texture':
        x = x*(blocksize-overlap)
        y = y*(blocksize-overlap)
        boundaries = skimage.segmentation.find_boundaries(best_cut, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        boundaries = skimage.segmentation.find_boundaries(boundaries, mode='thick').astype(bool)
        blur = cv2.GaussianBlur(New_pic,(15,15),0)
        condition1 = best_cut * ~boundaries
        condition2 = ~best_cut * ~boundaries
        condition3 = boundaries
        final_img[y:y+blocksize,x:x+blocksize,0] = condition1 * New_pic[:,:,0] + condition2 * final_img[y:y+blocksize,x:x+blocksize,0] + condition3  * blur[:,:,0]
        final_img[y:y+blocksize,x:x+blocksize,1] = condition1 * New_pic[:,:,1] + condition2 * final_img[y:y+blocksize,x:x+blocksize,1] + condition3  * blur[:,:,1]
        final_img[y:y+blocksize,x:x+blocksize,2] = condition1 * New_pic[:,:,2] + condition2 * final_img[y:y+blocksize,x:x+blocksize,2] + condition3  * blur[:,:,2]
        return 
    
def Find_Resemble_Texture():
    if flag == 'top marginal texture' or flag == 'first texture':
        result    = cv2.matchTemplate(Texture_img2,template,cv2.TM_CCORR_NORMED ) 
        threshold = np.max(result) * alpha
        loc = np.where( result >= threshold)
        n = np.random.randint(0,np.size(loc,axis=1),1)
        new_y = int(loc[0][n])
        new_x = int(loc[1][n])
        Texture_a = Texture_img[y_indx-blocksize//2:y_indx+blocksize//2+1,x_indx+blocksize//2-overlap+1:x_indx+blocksize//2+1].copy() 
        Texture_b = Texture_img[new_y:new_y+blocksize,new_x:new_x + overlap].copy()
        diff      = np.sum(np.abs(np.int16(Texture_a) - np.int16(Texture_b)),axis=2).T
        return   diff,new_y+blocksize//2,new_x+blocksize//2


    if flag == 'left marginal texture':
        result    = cv2.matchTemplate(Texture_img2,template,cv2.TM_CCORR_NORMED)
        threshold = np.max(result) * alpha
        loc = np.where( result >= threshold)
        n = np.random.randint(0,np.size(loc,axis=1),1)
        new_y = int(loc[0][n])  
        new_x = int(loc[1][n])
        Texture_a = Texture_img[y_indx+blocksize//2-overlap+1:y_indx+blocksize//2+1,x_indx-blocksize//2:x_indx+blocksize//2+1].copy() 
        Texture_b = Texture_img[new_y:new_y+ overlap,new_x:new_x+blocksize].copy()
        diff      = np.sum(np.abs(np.int16(Texture_a) - np.int16(Texture_b)),axis=2)
        return   diff,new_y+blocksize//2,new_x+blocksize//2
    
    
    if flag == 'middle marginal texture':
        result = np.zeros([rows,cols])
        result[blocksize//2:rows-blocksize//2,blocksize//2:cols-blocksize//2] = cv2.matchTemplate(Texture_img2,template,cv2.TM_CCORR_NORMED,None, mask)
        result[rows-blocksize//2-1]=0
        result[:,cols-blocksize//2-1]=0
        threshold = np.max(result) * alpha
        loc = np.where( result >= threshold)
        n = np.random.randint(0,np.size(loc,axis=1),1)
        new_y = int(loc[0][n])
        new_x = int(loc[1][n])
        Texture_a_h = template[0:overlap,0:blocksize]
        Texture_b_h = Texture_img[new_y-blocksize//2:new_y-blocksize//2+overlap,new_x-blocksize//2:new_x+blocksize//2+1].copy()
        Texture_a_v = template[0:blocksize,0:overlap]
        Texture_b_v = Texture_img[new_y-blocksize//2:new_y+blocksize//2+1,new_x-blocksize//2:new_x-blocksize//2+overlap].copy()
        horizontal_diff = np.sum(np.abs(np.int16(Texture_a_h) - np.int16(Texture_b_h)),axis=2)
        vertical_diff   = np.sum(np.abs(np.int16(Texture_a_v) - np.int16(Texture_b_v)),axis=2).T
        return vertical_diff,horizontal_diff,new_y,new_x
        
  
def Find_Best_Cut(diffrences):
    ways    = np.zeros([overlap,blocksize,1]).astype(np.uint16)
    E_first = np.zeros(overlap)
    E_last  = np.zeros(overlap)
    for j in range(blocksize-1):
        for i in range(overlap):
            if i==0:
                ways[i,j,0]  = np.where(diffrences[i:i+2,j]==np.min(diffrences[i:i+2,j]))[0][0] 
                E_first[i]   = np.min(diffrences[i:i+2,j]) + E_last[ways[i,j,0]] 
            elif i==blocksize-1 :
                ways[i,j,0] = np.where(diffrences[i-1:i+1,j]==np.min(diffrences[i-1:i+1,j]))[0][0] + i - 1
                E_first[i]  = np.min(diffrences[i-1:i+1,j]) + E_last[ways[i,j,0]]
            else:
                ways[i,j,0] = np.where(diffrences[i-1:i+2,j]==np.min(diffrences[i-1:i+2,j]))[0][0] + i - 1
                E_first[i]  = np.min(diffrences[i-1:i+2,j]) + E_last[ways[i,j,0]]
        E_last = np.copy(E_first)
    j = blocksize-1    
    for i in range(overlap):
        ways[i,j,0] = i
        E_first[i] = diffrences[i,j] + E_last[ways[i,j,0]]
    minimum_way = np.where(E_first==np.min(E_first))[0][0]
    Best_Cut = np.zeros([blocksize,1]).astype(np.int16)
    Best_Cut[-1] = ways[minimum_way,blocksize-1,0]
    for i in range(2,blocksize+1):
        Best_Cut[blocksize-i] = ways[Best_Cut[-i+1],-i,0] 
    return Best_Cut


start = time.time()
blocksize = 99
overlap   = 35
Texture_img  = cv2.imread('texture1.jpg')
Texture_img2 = Texture_img.copy()
Texture_img2[-blocksize::,:,:]=0
Texture_img2[:,-blocksize::,:]=0
[rows,cols]  = np.shape(Texture_img[:,:,0])
final_img    = np.zeros([2600,2600,3]).astype(np.uint8)
num =(2500 - blocksize) // (blocksize - overlap) + 1 
y_indx = np.random.randint(low = blocksize//2, high = rows-blocksize//2) 
x_indx = np.random.randint(low = blocksize//2, high = cols-blocksize//2) 
yy = y_indx 
xx = x_indx
New_pic = Texture_img[y_indx-blocksize//2:y_indx+blocksize//2+1,x_indx-blocksize//2:x_indx+blocksize//2+1]
y_loc = 1
x_loc = 1
best_cut = np.zeros([99,1])      
flag = 'first texture'
template = Fill_Image_Generate_Template(New_pic,y_loc,x_loc,flag)
vertical_idx     = np.indices(New_pic[:,:,0].shape)[0]
horizontal_idx     = np.indices(New_pic[:,:,0].shape)[1]
alpha = 1


flag = 'top marginal texture'
for counter1 in range(num):
    diff,y_indx,x_indx = Find_Resemble_Texture()
    best_cut = Find_Best_Cut(diff)
    best_cut = np.array(list(best_cut.T)*blocksize).T
    best_cut = best_cut <= horizontal_idx
    New_pic = Texture_img[y_indx-blocksize//2:y_indx+blocksize//2+1,x_indx-blocksize//2:x_indx+blocksize//2+1]
    template,x_loc = Fill_Image_Generate_Template(New_pic,y_loc,x_loc,flag)
    
    
y_indx = yy 
x_indx = xx
flag = 'left marginal texture'
template = final_img[blocksize-overlap:blocksize,0:blocksize]
for counter2 in range(num):
    diff,y_indx,x_indx = Find_Resemble_Texture()
    best_cut = Find_Best_Cut(diff)
    best_cut = np.array(list(best_cut.T)*blocksize)
    best_cut = best_cut <= vertical_idx
    New_pic = Texture_img[y_indx-blocksize//2:y_indx+blocksize//2+1,x_indx-blocksize//2:x_indx+blocksize//2+1]
    template,y_loc = Fill_Image_Generate_Template(New_pic,y_loc,x_loc,flag)
  
    
y_indx = yy 
x_indx = xx  
flag = 'middle marginal texture'
y_loc = 1
x_loc = 1
mask =np.zeros([blocksize,blocksize,3])
mask[0:blocksize,0:overlap] = 1
mask[0:overlap,overlap:blocksize] = 1
for counter1 in range(num):
    for counter2 in range(num):
        template = final_img[y_loc*(blocksize-overlap):y_loc*(blocksize-overlap)+blocksize,x_loc*(blocksize-overlap):x_loc*(blocksize-overlap)+blocksize]
        vertical_diff,horizontal_diff,y_indx,x_indx = Find_Resemble_Texture()
        best_cut_horizontal = Find_Best_Cut(horizontal_diff)
        best_cut_horizontal = np.array(list(best_cut_horizontal.T)*blocksize).T
        best_cut_horizontal = best_cut_horizontal <= horizontal_idx
        best_cut_vertical   = Find_Best_Cut(vertical_diff)
        best_cut_vertical   = np.array(list(best_cut_vertical.T)*blocksize)
        best_cut_vertical   = best_cut_vertical <= vertical_idx
        best_cut = best_cut_horizontal * best_cut_vertical
        New_pic = Texture_img[y_indx-blocksize//2:y_indx+blocksize//2+1,x_indx-blocksize//2:x_indx+blocksize//2+1]
        Fill_Image_Generate_Template(New_pic,y_loc,x_loc,flag)
        x_loc = x_loc + 1
    y_loc = y_loc + 1
    x_loc = 1


final_img = final_img[0:2500,0:2500,:]    
cv2.imwrite('Result.jpg',final_img)
end = time.time()
print(end - start)