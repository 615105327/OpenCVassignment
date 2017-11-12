import numpy as np
import cv2
import sys



def add_gaussian_noise(scrArr,mean,sigma):
	gauss = np.random.normal(mean,sigma,scrArr.shape)
	gauss = gauss.reshape(scrArr.shape)
	return np.add(scrArr, gauss, scrArr, casting='unsafe')

def add_salt_pepper(scrArr,pa,pb):
	out=np.copy(scrArr)
	coords = [np.random.randint(0, i - 1, int(pa*scrArr.size))
			for i in scrArr.shape]
	out[coords]=1
	coords = [np.random.randint(0, i - 1, int(pb*scrArr.size))
			for i in scrArr.shape]
	out[coords]=0
	return out


inputimage = cv2.imread(sys.argv[1])
cv2.imshow('Original',inputimage)
cv2.imwrite('Original.png',inputimage)

gaunoise = add_gaussian_noise(inputimage,0,50)
cv2.imshow('Gaussian Noise',gaunoise)
cv2.imwrite('Gaussian_noise.png',gaunoise)
boxfilt = cv2.blur(np.copy(gaunoise),(3,3))
cv2.imshow('Boxfilter', boxfilt)
cv2.imwrite('Gaussian_noise_box.png',boxfilt)
medianflit = cv2.medianBlur(np.copy(gaunoise),3)
cv2.imshow('medianBlur', medianflit)
cv2.imwrite('Gaussian_noise_med.png',medianflit)
gaussianfilt = cv2.GaussianBlur(np.copy(gaunoise),(3,3),1.5)
cv2.imshow('GaussianBlur', gaussianfilt)
cv2.imwrite('Gaussian_noise_gau.png',gaussianfilt)

saltnoise = add_salt_pepper(inputimage,0.01,0.01)
cv2.imwrite('salt_noise.png',saltnoise)
boxfilt = cv2.blur(np.copy(saltnoise),(3,3))
cv2.imshow('Box filter', boxfilt)
cv2.imwrite('Salt_noise_box.png',boxfilt)
medianflit = cv2.medianBlur(np.copy(saltnoise),3)
cv2.imshow('medianfilter', medianflit)
cv2.imwrite('Salt_noise_med.png',medianflit)
gaussianfilt = cv2.GaussianBlur(np.copy(saltnoise),(3,3),1.5)
cv2.imshow('Gaussianfilter', gaussianfilt)
cv2.imwrite('salt_noise_gau.png',gaussianfilt)

cv2.waitKey(0)

