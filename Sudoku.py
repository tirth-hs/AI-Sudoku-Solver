import cv2
import numpy as np
from PIL import Image 
from fastai.vision import *
from fastai.metrics import error_rate

classes = ['1','2','3','4','5','6','7','8','9']
path = './digit data new/'
np.random.seed(2)
data = ImageDataBunch.from_folder(path, train=".", valid_pct=0.2,
        ds_tfms=get_transforms(), size=224, num_workers=4).normalize(imagenet_stats)
learn = cnn_learner(data, models.resnet50, metrics=error_rate)
learn.load('tmp')

counter = 0
flag =0
maxarea = 0
maxcountour = [0]
cap=cv2.VideoCapture(0)

board=[]

for i in range (0,9):
	board.append([])
	for j in range(0,9):
		board[i].append(0)

def rescale_frame(frame, percent=75):
    width = int(frame.shape[1] * percent/ 100)
    height = int(frame.shape[0] * percent/ 100)
    dim = (width, height)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def checkrow(arr,row,num):
	for i in range(9):
		if(arr[row][i] == num):
			return False
	return True

def checkcol(arr,col,num):
	for i in range(9):
		if(arr[i][col] == num):
			return False
	return True

def checkbox(arr,row,col,num):
	for i in range(3):
		for j in range(3):
			if(arr[i+row][j+col] == num):
				return False
	return True

def safe(arr,row,col,num):
	return checkrow(arr,row,num) and checkcol(arr,col,num) and checkbox(arr,row - (row%3),col - (col%3),num)

def empty(arr,l):
	for row in range(9):
		for col in range(9):
			if(arr[row][col]==0):
				l[0]=row
				l[1]=col
				return False
	return True

def sudoku(arr):
	l = [0,0]
	if(empty(arr,l)):
		return True

	row = l[0]
	col = l[1]

	for num in range(1,10):
		if(safe(arr,row,col,num)):
			arr[row][col]=num	
			if(sudoku(arr)):
				return True
			arr[row][col] = 0
	return False


def solver(board):
	if(sudoku(board)):
		print(board)
	else:
		print("No solution exists")

def insertintob (board,k,no):
	count=1
	for i in range(9):
		for j in range(9):
			if count==k:
				if safe(board,i,j,no):
					board[i][j]=no
			count+=1
       
def arrange(cords):
	ar=[[0,0],[0,0],[0,0],[0,0]]
	max=0
	ymax=0
	min=11111111
	for i in range (len(cords)):
		if (cords[i][0][0]+cords[i][0][1])>max:
			max=cords[i][0][0]+cords[i][0][1]
			ar[2][0]=cords[i][0][0]
			ar[2][1]=cords[i][0][1]
		if (cords[i][0][0]+cords[i][0][1])<min:
			min=cords[i][0][0]+cords[i][0][1]
			ar[0][0]=cords[i][0][0]
			ar[0][1]=cords[i][0][1]
	for i in range (len(cords)):
		if (cords[i][0][0]+cords[i][0][1])<max and (cords[i][0][0]+cords[i][0][1])>min and cords[i][0][1]>ymax:
			ar[1][0]=cords[i][0][0]
			ar[1][1]=cords[i][0][1]
			ymax=cords[i][0][1]
	for i in range (len(cords)):
		if (cords[i][0][0]+cords[i][0][1])<max and (cords[i][0][0]+cords[i][0][1])>min and cords[i][0][1]<ymax:
			ar[3][0]=cords[i][0][0]
			ar[3][1]=cords[i][0][1]
	return ar
cords = None
while(1):
    _,frame=cap.read()
    img= cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    thresh=cv2.GaussianBlur(img,(3,3),0)
    thresh=cv2.adaptiveThreshold(thresh,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
    #cv2.imshow('thresh',thresh)
    contours,_=cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    maxarea = 0
    maxcountour = [0]
    for i in range(len(contours)):
        approx=cv2.approxPolyDP(contours[i],0.07*cv2.arcLength(contours[i],True),True)
        if cv2.contourArea(contours[i])>120000 and cv2.contourArea(contours[i]) <280000 and cv2.contourArea(contours[i])>maxarea and len(approx)==4:
            maxarea = cv2.contourArea(contours[i])
            maxcountour = approx
    if len(maxcountour) == 4:
        counter = counter+1
        cv2.drawContours(frame,[maxcountour],-1,(255,0,0),4)
        cv2.drawContours(frame,maxcountour,-1,(0,0,255),8)
    cv2.imshow('Solver',frame)
    if counter==4:
        cv2.imwrite("capture.jpg",frame)
        flag = 1
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    if flag==1:
        print("True")
        cords = maxcountour
        cap.release()
        break
cv2.destroyAllWindows()
maxarea = 0
maxcountour = [0]
cords = arrange(cords)
frame=cv2.imread('capture.jpg',1)
pts = np.float32([[0,0],[0,252],[252,252],[252,0]])
cords=np.float32(cords)
result = cv2.getPerspectiveTransform(cords,pts)
transform = cv2.warpPerspective(img,result,dsize=(252,252))
cv2.imwrite('warped.jpg',transform)
cv2.resize(transform,None,fx=2.5, fy=2.5, interpolation = cv2.INTER_CUBIC)
cv2.imshow('Question',transform)
k =1
for i in range(9):
	for j in range(9):
		crop=transform[(i*28):((i+1)*28),(j*28):((j+1)*28)]	
		crop=cv2.GaussianBlur(crop,(3,3),1)
		cv2.imwrite('Img{}.png'.format(k),crop)
		k=k+1
sum=0
count=0
lst=list()
digits=list()
for k in range(1,82):
	crop=cv2.imread('Img{}.png'.format(k),0)
	sum=0
	crop=cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,3,2)
	lst.append([k,crop])
	for i in range(10,19):
		for j in range(10,19):
			sum=sum+crop[i,j]
	j=81-(sum/255)
	if j>=10:
		count+=1
		digits.append(k)
print((digits))
print(len(digits))

image=np.empty([count,28,28,1])
grid=cv2.imread('warped.jpg',1)
for k in range(count):
	crop=cv2.imread('Img{}.png'.format(digits[k]),0)
	crop=cv2.adaptiveThreshold(crop,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,3,2)
	ret,crop=cv2.threshold(crop,0,255,cv2.THRESH_BINARY_INV)
	kernel=np.ones((1,1),np.uint8)
	crop=cv2.dilate(crop,kernel)
	crop=cv2.erode(crop,kernel)
	max2=[0]
	max=0
	crop = crop[6:23,6:23]
	crop1 = crop.copy()
	contours,hie=cv2.findContours(crop,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	for i in range(len(contours)):
		if cv2.contourArea(contours[i])>max:				
			max=cv2.contourArea(contours[i])
			max2=contours[i]
	cv2.drawContours(crop,[max2],-1,(255,255,255),-1)
	ret,thresh=cv2.threshold(crop,254,255,cv2.THRESH_BINARY)
	crop= cv2.bitwise_and(crop1,crop1,mask = thresh)
	ret,crop=cv2.threshold(crop,254,255,cv2.THRESH_BINARY_INV)
	crop=cv2.dilate(crop,kernel)
	crop=cv2.erode(crop,kernel)
	crop = cv2.bitwise_not(crop)
	cv2.imwrite('s{}.png'.format(k),crop)	
	if k == len(digits):
		break
	img = open_image('s{}.png'.format(k))
	pred_class,pred_idx,outputs = learn.predict(img)
	print(int(pred_class)+1)
	insertintob(board,digits[k],int(pred_class)+1)
solver(board)

final=cv2.imread('grid.png',1)
font = cv2.FONT_HERSHEY_DUPLEX
for i in range(9):
	for j in range (9):
		if(board[j][i] !=0):
			x=(i*28+8)
			y=((j)*28+20)
			value=str(board[j][i])
			cv2.putText(final,value,(x,y), font,0.5,(74,25,3),1,2)
			restext = cv2.resize(final,None,fx=2.5, fy=2.5, interpolation = cv2.INTER_CUBIC)
			cv2.imshow('final',restext)
			cv2.waitKey(50)
cv2.imwrite('solved.jpg',restext)

cv2.waitKey(0)
cv2.destroyAllWindows()

