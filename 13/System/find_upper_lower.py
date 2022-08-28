import cv2
import sys
import numpy as np

y = np.array([44,60,49,129,39,22,513,196,263,170,98,379,1600,2,31,5,1190,151,108,122,75,84,10,24,1375,7,1032,1429,1323,691])

def nothing(x):
    pass

useCamera=False

# Check if filename is passed
#if (len(sys.argv) <= 1) :
#    print("'Usage: python hsvThresholder.py <ImageFilePath>' to ignore camera and use a local image.")
#    useCamera = True

# Create a window
cv2.namedWindow('image', cv2.WINDOW_NORMAL)

# create trackbars for color change

cv2.createTrackbar('HMin','image',0,179,nothing) # Hue is from 0-179 for Opencv
cv2.createTrackbar('SMin','image',0,255,nothing)
cv2.createTrackbar('VMin','image',0,255,nothing)
cv2.createTrackbar('HMax','image',0,179,nothing)
cv2.createTrackbar('SMax','image',0,255,nothing)
cv2.createTrackbar('VMax','image',0,255,nothing)

HMin=179
SMin=255
VMin=255
HMax=179
SMax=255
VMax=255

HMin=17
SMin=77
VMin=153 # 콩의 경계선 부분이 없어짐
HMax=22
SMax=169 # 콩의 경계선 부분이 없어짐 (이걸 튜닝해야함)
VMax=218  # 콩의 중간 부분이 없어짐

# Set default value for MAX HSV trackbars.
cv2.setTrackbarPos('HMin', 'image', HMin)
cv2.setTrackbarPos('SMin', 'image', SMin)
cv2.setTrackbarPos('VMin', 'image', VMin)
cv2.setTrackbarPos('HMax', 'image', HMax)
cv2.setTrackbarPos('SMax', 'image', SMax)
cv2.setTrackbarPos('VMax', 'image', VMax)

# Initialize to check if HSV min/max value changes
hMin = sMin = vMin = hMax = sMax = vMax = 0
phMin = psMin = pvMin = phMax = psMax = pvMax = 0

sample_idx=1
img = cv2.imread(f'images/tuning/t{sample_idx}_5.jpg')
img = cv2.imread(f'images/tuning/t01_5.jpg')
output = img
waitTime = 33

while(True):

    # get current positions of all trackbars
    hMin = cv2.getTrackbarPos('HMin','image')
    sMin = cv2.getTrackbarPos('SMin','image')
    vMin = cv2.getTrackbarPos('VMin','image')

    hMax = cv2.getTrackbarPos('HMax','image')
    sMax = cv2.getTrackbarPos('SMax','image')
    vMax = cv2.getTrackbarPos('VMax','image')

    # Set minimum and max HSV values to display
    lower = np.array([hMin, sMin, vMin])
    upper = np.array([hMax, sMax, vMax])

    # Create HSV Image and threshold into a range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    output = cv2.bitwise_and(img,img, mask= mask)

    # Print if there is a change in HSV value
    if( (phMin != hMin) | (psMin != sMin) | (pvMin != vMin) | (phMax != hMax) | (psMax != sMax) | (pvMax != vMax) ):
        print("(hMin = %d , sMin = %d, vMin = %d), (hMax = %d , sMax = %d, vMax = %d)" % (hMin , sMin , vMin, hMax, sMax , vMax))
        phMin = hMin
        psMin = sMin
        pvMin = vMin
        phMax = hMax
        psMax = sMax
        pvMax = vMax

        

        

    # Display output image
    #dst2 = cv2.resize(output, dsize=(640, 480), fx=0.3, fy=0.7, interpolation=cv2.INTER_LINEAR)
    #resize = ResizeWithAspectRatio(image, width=1280)
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]


    bean_areas = []
    bean_count = 0

    for c in cnts:
        area = cv2.contourArea(c)
        #print(area)
        #print(area)
        if area > 1000:
            cv2.drawContours(output, [c], -1, (36, 255, 12), 2)
            bean_count = bean_count + int(area // 12000) + 1
            bean_areas.append(area)

    cv2.imshow('image',output)
    print(f'GT: {y[sample_idx-1]}, bean_count: {bean_count}, bean areas: {len(bean_areas)}')
    

    # Wait longer to prevent freeze for videos.
    if cv2.waitKey(waitTime) & 0xFF == ord('q'):
        print(f'len(bean areas): {len(bean_areas)}')
        print(f'bean areas: {(bean_areas)}')
        break

# Release resources
if useCamera:
    cap.release()
cv2.destroyAllWindows()

#[14263.0, 15229.0, 7231.5, 8868.5, 9341.5, 17184.5, 7379.0, 8577.5, 7395.5, 9655.0, 7569.0, 72932.0, 8672.5, 9016.5, 6823.5, 8738.5, 8137.5, 6718.0, 8847.0, 8051.5, 6652.5, 7586.5, 6774.5, 5255.0, 7245.5, 5452.5, 32262.5, 16506.0, 6715.5]
#[14263.0, 15229.0, 7231.5, 8868.5, 9341.5, 17184.5, 7379.0, 8577.5, 7395.5, 9655.0, 7569.0, 72932.0, 8672.5, 9016.5, 6823.5, 8738.5, 8137.5, 6718.0, 8847.0, 8051.5, 6652.5, 7586.5, 6774.5, 5255.0, 7245.5, 5452.5, 32263.5, 16506.0, 6715.5]
#[14244.0, 15068.5, 7231.5, 8868.5, 9341.5, 17180.5, 7351.5, 8577.5, 8247.0, 14027.5, 15764.0, 7395.5, 9629.0, 7569.0, 28787.0, 8641.5, 9015.0, 6823.5, 8735.0, 8133.5, 6609.0, 8847.0, 8029.0, 6652.5, 7416.5, 6755.5, 5247.5, 23026.5, 7232.5, 5397.0, 8280.5, 16412.0, 6643.5]