from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("Syngenta.bmp")

cores = image.getcolors

print(cores)# o verde esta em 58 do RGB ou seja eu posso contar quais são iguais a 58 pixels
pixels = list(image.getdata())
int tam = pixels.count
cont = 0
print(tam)

#for x in tam:
#    if tam[x] == 58:
#        cont +=1
        







#total = len(list(filter(lambda i: i == (50,205,50), pixels)))
#print("There are %d bright pixels" % total)
#plt.imshow(image)
#print(image.size) #420 largura x 300 altura
#print(image.format) #BMP
#print(image.mode) #P 