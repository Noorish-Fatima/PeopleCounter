
from ultralytics import YOLO
import cv2
import cvzone 
import time
import math
from IPython.display import display, Image, clear_output
from sort import *

cap=cv2.VideoCapture("Videos/people.mp4")

model=YOLO("./YoloWeights/yolov8n.pt")

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"
              ]

mask=cv2.imread("Project/PeopleCounter/PeopleMask.png")

# tracking
tracker=Sort(max_age=20,min_hits=2,iou_threshold=0.3)
limitsUp=[180,480,500,410]
limitsDown=[760,582,980,520]
totalCountUp=[]
totalCountDown=[]

while True:
    success,img=cap.read()
    imgRegion=cv2.bitwise_and(img,mask)

    imgGraphic=cv2.imread("Project/PeopleCounter/graphics.png",cv2.IMREAD_UNCHANGED)
    #if imgGraphic.shape[2] == 3:
        #imgGraphic = cv2.cvtColor(imgGraphic, cv2.COLOR_BGR2BGRA)
    img=cvzone.overlayPNG(img,imgGraphic,(1370, 360))
    results=model(imgRegion)
    detections=np.empty((0,5))
    for r in results:
        boxes=r.boxes
        for box in boxes:
            # OpenCV

            #bounding box
            x1,y1,x2,y2=box.xyxy[0]
            x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
            #print(x1,y1,x2,y2)
            #cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)

            # cvzone
            w,h=x2-x1,y2-y1
            #confidence
            conf=math.ceil((box.conf[0]*100))/100
            #cvzone.putTextRect(img,f'{conf}',(max(0,x1),max(35,y1)))

            #class name
            cls=int(box.cls[0])
            currentClass=classNames[cls]

            if currentClass=='person' and conf>0.3:
                #cvzone.putTextRect(img,f'{currentClass} {conf}',(max(0,x1),max(35,y1)),scale=4,thickness=2,offset=3)
                #cvzone.cornerRect(img,(x1,y1,w,h),l=15,rt=5)

                currentArray=np.array([x1,y1,x2,y2,conf])
                detections=np.vstack((detections,currentArray))

    resultsTracker=tracker.update(detections)
    cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,0,255),10)
    cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,0,255),10)

    for result in resultsTracker:
        x1,y1,x2,y2,id=result
        x1,y1,x2,y2=int(x1),int(y1),int(x2),int(y2)
        print(result)
        w,h=x2-x1,y2-y1
        cvzone.cornerRect(img,(x1,y1,w,h),l=15,rt=2,colorR=(255,255,0))
        cvzone.putTextRect(img,f'{int(id)}',(max(0,x1),max(35,y1)),scale=3,thickness=3,offset=6)

        cx,cy=x1+w//2,y1+h//2
        cv2.circle(img,(cx,cy),5,(0,0,255),cv2.FILLED)

        if limitsUp[0]<cx<limitsUp[2] and limitsUp[1]-35<cy<limitsUp[1]+35:
            if totalCountUp.count(id)==0:
                totalCountUp.append(id)
                cv2.line(img,(limitsUp[0],limitsUp[1]),(limitsUp[2],limitsUp[3]),(0,255,0),10)


        if limitsDown[0]<cx<limitsDown[2] and limitsDown[1]-10<cy<limitsDown[1]+10:
            if totalCountDown.count(id)==0:
                totalCountDown.append(id)
                cv2.line(img,(limitsDown[0],limitsDown[1]),(limitsDown[2],limitsDown[3]),(0,255,0),10)

        cv2.putText(img,f'{len(totalCountUp)}',(1560,453),fontScale=6,fontFace=1,color=(0,255,30),thickness=8)
        cv2.putText(img,f'{len(totalCountDown)}',(1810,452),fontScale=6,fontFace=1,color=(0,0,255),thickness=8)

    _, encoded_img = cv2.imencode('.jpg', img)
    cv2.imshow("image",img)
    clear_output(wait=True)
    display(Image(data=encoded_img))
    #cv2.imshow("region",imgRegion)
    if cv2.waitKey(1)&0xFF==ord('q'):
        break 

cap.release()
cv2.destroyAllWindows()