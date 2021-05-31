import cv2 
from matplotlib import pyplot as plt

img = cv2.imread("Syngenta.bmp")

img_color = cv2.cvtColor(img, cv2.COLOR_RGB2HSV_FULL)

ret, thresh = cv2.threshold(img_color, 150, 255, cv2.THRESH_BINARY_INV)
#Alterando a coloração de fundo para branco para ver se encontro

plt.imshow(thresh)

plt.show()
