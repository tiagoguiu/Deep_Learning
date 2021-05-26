from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

image = Image.open("Syngenta.bmp")
pixels = list(image.getdata())
tam2 = pixels.index(51) #procurando o verde dentro da imagem pelo index RGB
print(tam2) #QUANTIDADE TOTAL DE PONTOS VERDES 481

#DEIXAREI COMENTADO O QUE EU TENTEI FAZER

#print(tam)
#for x in tam:
#    if tam[x] == 58:
#        cont +=1
#total = len(list(filter(lambda i: i == (50,205,50), pixels)))
#print("There are %d bright pixels" % total)
#plt.imshow(image)
#print(image.size) #420 largura x 300 altura
#print(image.format) #BMP
#print(image.mode) #P 

#SEGUNDA PARTE DA ATIVIDADE



thresh = 200
fn = lambda x : 255 if x > thresh else 0
r = image.convert('L').point(fn, mode='1')#nada achado em transformado tudo em branco
r.show()