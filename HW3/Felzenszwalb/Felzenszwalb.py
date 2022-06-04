import  numpy as np
import  cv2
from skimage.segmentation import felzenszwalb
import skimage

def add_point(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        x_points.append(x)
        y_points.append(y)
        
img = cv2.imread('birds.jpg')
image_b = cv2.resize(img[:,:,0], (1600,1200), interpolation = cv2.INTER_AREA)
image_g = cv2.resize(img[:,:,1], (1600,1200), interpolation = cv2.INTER_AREA)
image_r = cv2.resize(img[:,:,2], (1600,1200), interpolation = cv2.INTER_AREA)
img = np.zeros([1200,1600,3])
img[:,:,0] = image_b 
img[:,:,1] = image_g
img[:,:,2] = image_r
img = np.uint8(img)
cv2.namedWindow("birds",cv2.WINDOW_NORMAL)
x_points = []
y_points = []
cv2.setMouseCallback("birds", add_point)
while True: 
    cv2.imshow("birds",img)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()  
x_points = np.array(x_points)
y_points = np.array(y_points)   
n = len(x_points)
segments = felzenszwalb(img, scale=500, sigma=0.95, min_size=20)
number_of_clusters = np.max(segments)
   
final_im  = np.zeros_like(img)    
for i in range(n):
    final_im[:,:,0] += ((segments == segments[y_points[i],x_points[i]]) * img[:,:,0]) * (final_im[:,:,0]==0)
    final_im[:,:,1] += ((segments == segments[y_points[i],x_points[i]]) * img[:,:,1]) * (final_im[:,:,1]==0)
    final_im[:,:,2] += ((segments == segments[y_points[i],x_points[i]]) * img[:,:,2]) * (final_im[:,:,2]==0)

cv2.imwrite('Result.jpg',final_im)