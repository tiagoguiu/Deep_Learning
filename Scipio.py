
import matplotlib.pyplot as plt
import cv2
import numpy as np
from PIL import Image


#abrindo imagem
image = Image.open("meteor_challenge_01.png")

#pegando os pixels da imagem em RGBA
pixels = list(image.getdata())

#defindo cor de busca
cor = (255,0,0,255)

#buscando dentro dos pixels quais são vermelho #ff0000 ou pure red
tam2 = pixels.count(cor)

#printando
print(tam2)

#definindo cor branca ou pure white #fffffff
cor2 = (255,255,255,255)

#bucando dentro dos pixels quais são brancos
tam = pixels.count(cor2)

#printando
print(tam)


 