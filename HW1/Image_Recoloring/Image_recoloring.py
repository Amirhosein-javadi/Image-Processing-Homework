import numpy as np
import cv2
yellow = cv2.imread('Image2.jpg')

R_1 = yellow[:,:,2].astype(float)
G_1 = yellow[:,:,1].astype(float)
B_1 = yellow[:,:,0].astype(float)
new_pic=np.copy(yellow)
tr1=R_1-B_1
tr2=G_1-B_1

statement1 = (R_1>210) * (G_1>210) * (B_1<190)
statement2 = (tr1>85)  * (tr2>53)
           
new_pic[:,:,1] = np.floor(255*(((G_1[:,:]-50)/255)**10)) * (statement1)
new_pic[:,:,0] = np.floor(255*(((B_1[:,:]-50)/255)**2)) * (statement1)
new_pic[:,:,1] = np.floor(255*((G_1[:,:]/255)**8)) * (~statement1) * (statement2)
new_pic[:,:,0] = np.floor(255*((B_1[:,:]/255)**1.1)) * (~statement1) * (statement2)
new_pic[:,:,1] = G_1 * (~statement1) * (~statement2)
new_pic[:,:,0] = B_1 * (~statement1) * (~statement2)

cv2.imwrite('Image2_recolor.jpg', new_pic) 

# part 2 
pink = cv2.imread('Image1.jpg')     
R_2 = np.copy(pink[:,:,2].astype(float))
G_2 = np.copy(pink[:,:,1].astype(float))
B_2 = np.copy(pink[:,:,0].astype(float))
Cmax= (R_2>=G_2) * (R_2>B_2) * R_2 + (G_2>R_2) * (G_2>=B_2) * G_2 + (B_2>=R_2) * (B_2>G_2) * B_2 + (B_2==G_2) * (B_2==R_2) * G_2
Cmin= (R_2<=G_2) * (R_2<B_2) * R_2 + (G_2<R_2) * (G_2<=B_2) * G_2 + (B_2<=R_2) * (B_2<G_2) * B_2 + (B_2==G_2) * (B_2==R_2) * G_2
delta=Cmax-Cmin
delta -= (delta==0) *1
H = (Cmax==R_2) * (((G_2-B_2)/delta)%6)*60 +\
    (Cmax!=R_2) * (Cmax==G_2) * (((B_2-R_2)/delta)+2)*60 +\
    (Cmax!=R_2) * (Cmax!=G_2) * (Cmax==B_2) * (((R_2-G_2)/delta)+4)*60
delta += (delta==-1) *1
S = delta/Cmax
H -= (H>=300)* (S>0.20) *100
V = np.copy(Cmax/255)
C = V * S
X = C * (1 - abs((H / 60) % 2 - 1))
M = V - C
new_pic2=np.zeros([4032,3024,3]).astype(np.uint8)
R_prim= (H<60)  * C + (60<=H)*(H<120)  * X + (120<=H)*(H<240) * 0 + (240<=H)*(H<300) * X + (300<=H)*(H<360) * C
G_prim= (H<60)  * X + (60<=H)*(H<120)  * C + (120<=H)*(H<180) * C + (180<=H)*(H<240) * X + (240<=H)*(H<360) * 0
B_prim= (H<120) * 0 + (120<=H)*(H<180) * X + (180<=H)*(H<240) * C + (240<=H)*(H<300) * C + (300<=H)*(H<360) * X

new_pic2[:,:,0]=(B_prim+M)*255
new_pic2[:,:,1]=(G_prim+M)*255
new_pic2[:,:,2]=(R_prim+M)*255

cv2.imwrite('Image1_recolor.jpg',new_pic2) 