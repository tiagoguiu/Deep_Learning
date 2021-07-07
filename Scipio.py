
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image



red = [(255,70,70),(255,0,0)] 
white = [(170,170,170),(255,255,255)]
yellow = [(0,240,250),(10,255,255)]
dot_colors = [red, white, yellow]
    
img = cv2.imread('meteor_challenge_01.png')   
# aplicando medianBlur para suavizar a imagem antes do limiar 
blur= cv2.medianBlur(img, 7) 

for lower, upper in dot_colors:
    output = img.copy()
    # aplicando a cor limite ao branco (255,255, 255) e o resto ao preto (0,0,0)
    mask = cv2.inRange(blur,lower,upper) 

    circles = cv2.HoughCircles(mask,cv2.HOUGH_GRADIENT,1,20,param1=20,param2=8,
                               minRadius=0,maxRadius=60)    
    index = 0
    if circles is not None:
        # convertendo as coordenadas (x, y) e o raio dos círculos em inteiros
        circles = np.round(circles[0, :]).astype("int")

        # fazendo um loop sobre as coordenadas (x, y) e o raio dos círculos
        for (x, y, r) in circles:
            # desenhando o círculo na imagem de saída, 
            # desenhando um retângulo correspondente ao centro do círculo
            cv2.circle(output, (x, y), r, (255, 0, 255), 2)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (255, 0, 255), -1)

            index = index + 1
            print (str(index) + " : " + str(r) + ", (x,y) = " + str(x) + ', ' + str(y))
        print ('No. of circles detected = {}'.format(index))
print(output)
