import numpy
import cv2
import sys

origininput = cv2.imread(sys.argv[1])
[b,g,r] = cv2.split(origininput)
cv2.imwrite('Blue.png',b)
cv2.imwrite('Green.png',g)
cv2.imwrite('Red.png',r)
cv2.imshow('Blue',b)
cv2.imshow('Green',g)
cv2.imshow('Red',r)
print("RGB: ",origininput[20,25],)
print("Blue from ",b.max(),"to ",b.min())
print("Green from ",g.max(),"to ",g.min())
print("Red from ",r.max(),"to ",r.min())

ycrcbimage = cv2.cvtColor(origininput, cv2.COLOR_BGR2YCR_CB)
[y,cb,cr] = cv2.split(ycrcbimage)
cv2.imwrite('Y.png',y)
cv2.imwrite('Cb.png',cb)
cv2.imwrite('Cr.png',cr)
cv2.imshow('Y',y)
cv2.imshow('Cb',cb)
cv2.imshow('Cr',cr)
print("YCbCr: ",ycrcbimage[20,25])
print("Y from ",y.max(),"to ",y.min())
print("Cb from ",cb.max(),"to ",cb.min())
print("Cr from ",cr.max(),"to ",cr.min())

hsvimage = cv2.cvtColor(origininput, cv2.COLOR_BGR2HSV)
[h,s,v] = cv2.split(hsvimage)
cv2.imwrite('Hue.png',h)
cv2.imwrite('Saturation.png',s)
cv2.imwrite('Value.png',v)
cv2.imshow('Hue',h)
cv2.imshow('Saturation',s)
cv2.imshow('Value',v)
print("HSV: ",hsvimage[20,25])
print("Yue from ",h.max(),"to ",h.min())
print("Saturation from ",s.max(),"to ",s.min())
print("Value from ",v.max(),"to ",v.min())

cv2.waitKey(0)
