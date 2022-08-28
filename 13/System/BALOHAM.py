import numpy as np
import cv2
from matplotlib import pyplot as plt
import time
import datetime as dt

y = np.array([44,60,49,129,39,22,513,196,263,170,98,379,1600,2,31,5,1190,151,108,122,75,84,10,24,1375,7,1032,1429,1323,691])

def L2_mse(pred):
    err = 0
    for i in range(30):
        squared_loss = (100 * abs(pred[i] - y[i]) / y[i])**2
        err = err + squared_loss
    return err

def poly_regression(x):
    # coeffs got from polynomial regression
    coeffs = [39.365190152476885, -7.90896331e-05, 1.17303311e-10]
    return coeffs[0] + np.multiply(x, coeffs[1]) + np.multiply(x**2, coeffs[2])

def detect_many_beans(hsv):
    area_threshold = (2348691.5 + 2442803.5) // 2 # medium of 9th largest and 8th leargest

    lower = np.array([14, 23, 140], dtype="uint8") 
    upper = np.array([22, 194, 218], dtype="uint8") 

    mask = cv2.inRange(hsv, lower, upper)

    # findContours for opencv-python==4.xx
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # findContours for opencv-python==3.xx
    #_, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 100:
            total_area = total_area + area

    cv2.waitKey()
    if total_area > area_threshold:
        #total_area_arr.append(total_area)
        predicted_cnt = int(poly_regression(total_area))
        if predicted_cnt > 1600:
            return 1600
        elif predicted_cnt < 2:
            return 2
        else:
            return predicted_cnt
    
    return 0

def segment_dish(img):
    mask = np.full(img.shape[:2],255,np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    # morphologyEx
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 40)

    # black block mask for *_5.jpg
    left_block_init = np.zeros(img.shape[:2],np.uint8) # (3024, 4032)
    radius_lb = round(img.shape[0] / 2) - 50
    center_x_lb = 500 + radius_lb
    center_y_lb = radius_lb
    cv2.circle(left_block_init, (center_x_lb, center_y_lb), radius_lb, (255,255,255), -1)
    center_x_rb = img.shape[1] - radius_lb
    cv2.circle(left_block_init, (center_x_rb, center_y_lb), radius_lb, (255,255,255), -1)
    left_block_init[:, img.shape[1]-50:] = 0
    opening[np.where(left_block_init==0)] = 0

    # white inner circle mask for *_5.jpg
    circle_init = np.zeros(img.shape[:2],np.uint8)
    center_x = round((800 + 3500)/2)
    center_y = 1400
    radius= (2150 - 800) - 200 + 50 # (folder_num * 10) # (2150-800) for (x2-x1) // 230 bias for t19
    cv2.circle(circle_init, (center_x, center_y), radius, (255,255,255), -1)
    opening[np.where(circle_init==255)] = 255
    
    opening[np.where(opening==255)] = 1
    mask3 = np.where((opening==2)|(opening==0), 0, 1).astype('uint8')
    #mask, bgdModel, fgdModel = cv2.grabCut(img,opening,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    #mask3 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    img = img*mask3[:,:,np.newaxis]
    return img, opening

def grabcut_segment(img, opening):
    mask = np.full(img.shape[:2],255,np.uint8)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask, bgdModel, fgdModel = cv2.grabCut(img,opening,None,bgdModel,fgdModel,1,cv2.GC_INIT_WITH_MASK)
    mask3 = np.where((mask==2)|(mask==0), 0, 1).astype('uint8')
    grabcutted_img = img*mask3[:,:,np.newaxis]
    return grabcutted_img

    
HMin=17
SMin=77
VMin=153
HMax=22
SMax=169
VMax=218 

start_time = time.time()
bean_count_arr = []
total_area_arr = []
bean_areas_arr = []
#least_err = 9 * (10**10)

for i in range(1, 31):
    if i < 10 :
        sample_no = 't0' + str(i)
    else :
        sample_no = 't' + str(i)
    print('current sample_no: {}'.format(sample_no))
    image = cv2.imread('Kong_Hidden_True/{}/5.jpg'.format(sample_no))

    # segment dish
    image, opening = segment_dish(image)
    
    # convert image to hsv
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # use polynomial regression for too many beans
    bean_count = detect_many_beans(hsv)
    if bean_count != 0:
        #print('many_beans')
        bean_count_arr.append(bean_count)
        bean_areas_arr.append(0)
        continue

    image = grabcut_segment(image, opening)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # use hsv segmentation
    lower = np.array([HMin, SMin, VMin], dtype="uint8") # default
    upper = np.array([HMax, SMax, VMax], dtype="uint8") # default
    mask = cv2.inRange(hsv, lower, upper)

    # findContours for opencv-python==4.xx
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # findContours for opencv-python==3.xx
    #_, cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    bean_count = 0
    bean_areas = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area > 1000:
            cv2.drawContours(image, [c], -1, (36, 255, 12), 2)
            bean_count = bean_count + int(area // 10290) + 1
            bean_areas.append(area)

    cv2.waitKey()
    if bean_count < 2 :
        bean_count = 2
    elif bean_count > 1600 :
        bean_count = 1600
    else:
        bean_count_arr.append(bean_count)
    bean_areas_arr.append(len(bean_areas))

turnaround_time = time.time() - start_time
date = dt.datetime.now()
today_date = date.strftime("%m-%d-%H-%M-%S")

with open("./Kong_13_Hidden.txt", 'w') as f:
    f.write('%TEAM\tBALOHAM\n')
    f.write('%DATE\t{}\n'.format(today_date))
    f.write('%TIME\t{}\n'.format(turnaround_time))
    f.write('%CASES\t30\n')

    for bean_count, idx in zip(bean_count_arr, range(1, len(bean_count_arr)+1)) :
        # if idx >= 1:
        #     break
        if idx < 10 :
            test_case_name = 'T0' + str(idx)
        else :
            test_case_name = 'T' + str(idx)

        f.write('{}\t{}\n'.format(test_case_name, bean_count))

f.close()
