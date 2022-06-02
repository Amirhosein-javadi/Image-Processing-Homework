import  numpy as np
import  cv2


flowers = cv2.imread('flowers_blur.png').astype(np.float)
filter_size = 41
sigma_filter = 15
my_filter = np.zeros([filter_size,filter_size])

for i in range(filter_size):
    for j in range(filter_size):
        my_filter[j,i] = np.exp(-((j-filter_size//2)**2+(i-filter_size//2)**2)/(2*sigma_filter**2))

my_filter = my_filter / np.sum(my_filter)
unsharp_filter = - my_filter
unsharp_filter[filter_size//2,filter_size//2] += 1
flowers_unsharp =  cv2.filter2D(flowers,-1,unsharp_filter)
cv2.imwrite("res01.jpg", flowers_unsharp)

flowers_Sharp= flowers + 0.5 * flowers_unsharp
cv2.imwrite("Sharp_Img.jpg", flowers_Sharp)