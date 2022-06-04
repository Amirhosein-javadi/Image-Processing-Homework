import  numpy as np
import  cv2
import matplotlib.pyplot as plt

text = open("Points.txt","r")
Points =text.readlines()
number = int(Points[0])
x = np.zeros([number,1])
y = np.zeros([number,1])

for i in range(number):
    x[i]= float(Points[i+1].split()[0])
    y[i]= float(Points[i+1].split()[1])

plt.plot( y, x,'o', 'b')
plt.savefig('res01.jpg')    
plt.clf()

Coordinates = np.float32(np.hstack((x,y)))
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1)
compactness,labels,centers = cv2.kmeans(Coordinates,2,None,criteria,200,cv2.KMEANS_RANDOM_CENTERS)

cluster_A = Coordinates[labels.ravel()==0]
cluster_B = Coordinates[labels.ravel()==1]
plt.plot(cluster_A[:,1], cluster_A[:,0],'o', 'b')
plt.plot(cluster_B[:,1], cluster_B[:,0],'o', 'r')
plt.savefig('res02.jpg')   
plt.clf()


r = np.float32(x**2 + y**2)
compactness_prim,labels_prim,centers_prim = cv2.kmeans(r,2,None,criteria,10,cv2.KMEANS_RANDOM_CENTERS)
cluster_A_prim = np.vstack((x[labels_prim==0],y[labels_prim==0])).T 
cluster_B_prim = np.vstack((x[labels_prim==1],y[labels_prim==1])).T
plt.plot(cluster_A_prim[:,1], cluster_A_prim[:,0],'o', 'b')
plt.plot(cluster_B_prim[:,1], cluster_B_prim[:,0],'o', 'r')
plt.savefig('res03.jpg')   
plt.clf()



