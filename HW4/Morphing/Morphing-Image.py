import  numpy as np
import  cv2
from scipy.spatial import Delaunay
import os
import time

# def add_point(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDBLCLK:
#         x_points.append(x)
#         y_points.append(y)

# Obama = cv2.imread('Obama_spoted.jpg')
# cv2.namedWindow("Obama",cv2.WINDOW_NORMAL)
# x_points = []
# y_points = []
# cv2.setMouseCallback("Obama", add_point)
# while True: 
#     cv2.imshow("Obama",Obama)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break   
# cv2.destroyAllWindows()   
# x_points = np.int16(x_points)
# y_points = np.int16(y_points)
# Obama= open("Obama.txt","a")
# for i in range(len(x_points)):
#     Obama.write(str(y_points[i]) + " " + str(x_points[i]) + "\n")
# Obama.close()

# Biden = cv2.imread('Biden_spoted.jpg') 
# cv2.namedWindow("Biden",cv2.WINDOW_NORMAL)
# x_points = []
# y_points = []
# cv2.setMouseCallback("Biden", add_point)
# while True: 
#     cv2.imshow("Biden",Biden)
#     if cv2.waitKey(1) & 0xFF == ord("q"):
#         break   
# cv2.destroyAllWindows()   
# x_points = np.int16(x_points)
# y_points = np.int16(y_points)
# Biden= open("Biden.txt","a")
# for i in range(len(x_points)):
#     Biden.write(str(y_points[i]) + " " + str(x_points[i]) + "\n")
# Biden.close()
start = time.time()   
Obama = cv2.imread('Obama.jpg')
Biden = cv2.imread('Biden.jpg') 
Obama_Txt= open("Obama.txt","r")
Points = Obama_Txt.readlines()
number = len(Points)
x1 = np.zeros([number]).astype(np.int16)
y1 = np.zeros([number]).astype(np.int16)
for i in range(number):
    y1[i]= int(Points[i].split()[0])
    x1[i]= int(Points[i].split()[1])
Obama_Txt.close()

Biden_Txt= open("Biden.txt","r")
Points = Biden_Txt.readlines()
x2 = np.zeros([number]).astype(np.int16)
y2 = np.zeros([number]).astype(np.int16)
for i in range(number):
    y2[i]= int(Points[i].split()[0])
    x2[i]= int(Points[i].split()[1])
Biden_Txt.close()

delta_x = x2 - x1
delta_y = y2 - y1
num = 50                           # number of frames
points = np.zeros([number,2])
points[:,0] = x1
points[:,1] = y1
tri = Delaunay(points)
segments = tri.simplices
num_of_sgmnt = np.size(segments,axis = 0)
final_im = np.zeros_like(Biden)
warp_dst_1 = np.zeros_like(Biden)
warp_dst_2 = np.zeros_like(Biden)
pts_src1  = np.zeros([3,2,num_of_sgmnt]).astype(np.float32)
pts_src2  = np.zeros([3,2,num_of_sgmnt]).astype(np.float32)
shape = (warp_dst_1.shape[1], warp_dst_1.shape[0])
mask      = np.zeros_like(Obama)


for counter2 in range(num_of_sgmnt):
    pts_src1[:,:,counter2]  = np.float32([[x1[segments[counter2,0]],y1[segments[counter2,0]]], [x1[segments[counter2,1]],y1[segments[counter2,1]]], [x1[segments[counter2,2]],y1[segments[counter2,2]]]])
    pts_src2[:,:,counter2]  = np.float32([[x2[segments[counter2,0]],y2[segments[counter2,0]]], [x2[segments[counter2,1]],y2[segments[counter2,1]]], [x2[segments[counter2,2]],y2[segments[counter2,2]]]])
    
    
for counter1 in range(1,num+1):
    x = np.uint16((x1 + (delta_x/num) * counter1))
    y = np.uint16((y1 + (delta_y/num) * counter1))
    for counter2 in range(num_of_sgmnt):
        pts_dst   = np.float32([[x [segments[counter2,0]],y [segments[counter2,0]]], [x [segments[counter2,1]],y [segments[counter2,1]]], [x [segments[counter2,2]],y [segments[counter2,2]]]])
        warp_mat1 = cv2.getAffineTransform(pts_src1[:,:,counter2],pts_dst)
        warp_mat2 = cv2.getAffineTransform(pts_src2[:,:,counter2],pts_dst)
        triangle  = np.int32(pts_dst)
        cv2.fillConvexPoly(mask, triangle, (1, 1, 1))
        warp_dst_1 = np.int16(cv2.warpAffine(Obama, warp_mat1, shape))
        warp_dst_2 = np.int16(cv2.warpAffine(Biden, warp_mat2, shape))
        state      = final_im == 0
        final_im   = final_im + (warp_dst_1*(num-counter1) + warp_dst_2*(counter1))/num * mask * state
        mask[:,:,:] = 0
    filename = "pics/file-%d.png"%(counter1)
    cv2.imwrite(filename, final_im)
    final_im = 0  
        
end = time.time()
print(end - start)
os.system('ffmpeg -i pics/file-%d.png -r 15 -vcodec mpeg4 Result.MP4')