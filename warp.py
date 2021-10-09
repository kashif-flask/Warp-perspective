import cv2
import numpy as np


def processing(img):
    imggray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #cv2.imshow("imagegray",imggray)
    #imghsv=cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    #cv2.imshow("imagehsv",imghsv)
    #imgcanny=cv2.Canny(imggray,200,200)
    #cv2.imshow("imagecanny",imgcanny)
    imgblur=cv2.GaussianBlur(imggray,(5,5),1)
    #cv2.imshow("imageblur",imgblur)
    imgcannyblur=cv2.Canny(imgblur,200,200)
    #cv2.imshow("imagecannyblur",imgcannyblur)
    kernel=np.ones((5,5))
    imgdilate=cv2.dilate(imgcannyblur,kernel,iterations=2)
    #cv2.imshow("imgdilate",imgdilate)
    imgerode=cv2.erode(imgdilate,kernel,iterations=1)
   # cv2.imshow("imageerode",imgerode)
    return imgerode

def getcontour(imgerode,img2):
    contours,hierarchy=cv2.findContours(imgerode,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    high=0
    biggest=np.array([])
    for cnt in contours:
        area=cv2.contourArea(cnt)
        
        if area>5000:
            #print(area)
            #cv2.drawContours(img2,cnt,-1,(0,255,0),3)
            peri=cv2.arcLength(cnt,True)
            approx=cv2.approxPolyDP(cnt,0.02*peri,True)
            if area>high and len(approx)==4:
                biggest=approx
                high=area
    cv2.drawContours(img2,biggest,-1,(0,255,0),20)
    return biggest
   
#print(high)
#print(x,y,w,h)
#cv2.rectangle(img2,(x,y),(x+w,y+h),(0,255,0),2)
#print(len(approx))
def reorder(points):
    points=points.reshape((4,2))
    newpoints=np.zeros((4,1,2),np.int32)
    arrsum=points.sum(axis=1)
    newpoints[0]=points[np.argmin(arrsum)]
    newpoints[3]=points[np.argmax(arrsum)]
    arrdif=np.diff(points,axis=1)
    newpoints[1]=points[np.argmin(arrdif)]
    newpoints[2]=points[np.argmax(arrdif)]
    return newpoints




#print(newpoints)
#img=cv2.imread("card1.png")
#img=cv2.resize(img,(0,0),None,0.4,0.4)
cap=cv2.VideoCapture("vid1.mp4") #you can use webcam also

while True:
    success,img=cap.read()
    if success:
        img=cv2.resize(img,(0,0),None,0.4,0.4)
        img2=img.copy()
        #cv2.imshow("img",img)
        size=img.shape
        imgerode=processing(img)
        biggest=getcontour(imgerode,img2)
        #cv2.imshow("contour",img2)
        if biggest.size>0:
            newpoints=reorder(biggest)
            newpoints=np.float32(newpoints).reshape((4,2))
            pts2=[[0,0],[size[1],0],[0,size[0]],[size[1],size[0]]]
            pts2=np.float32(pts2)
            matrix=cv2.getPerspectiveTransform(newpoints,pts2)
            imgwarp=cv2.warpPerspective(img,matrix,(size[1],size[0]))
            imgwarp_rotated=cv2.rotate(imgwarp,cv2.cv2.ROTATE_90_CLOCKWISE)
            gray_warp=cv2.cvtColor(imgwarp_rotated,cv2.COLOR_BGR2GRAY)
            mask=cv2.adaptiveThreshold(gray_warp,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,7,2)
            #mask_blur=cv2.medianBlur(mask,5)
            cv2.imshow("imgwarp",imgwarp_rotated)
            cv2.imshow("mask",mask)
            
            

        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
    else:
        break
    
cap.release()
cv2.destroyAllWindows
