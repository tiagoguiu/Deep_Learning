import math

#QUESTAO 1
print("Digite a entrada")

a = int(input())

if a == 31:
    print("Belo Horizonte")
elif a == 61:
    print("Brasilia")
elif a == 71:
    print("Salvador")
elif a == 11:
    print("São Paulo")
elif a == 21:
    print("Rio de Janeiro")
elif a == 32:
    print("Juiz de Fora")
elif a == 19:
    print("Campinas")
elif a == 27:
    print("Vitória")

#QUESTÃO 2
vl = float(input())

tx = float(input())

mes = int(input())

if vl > 0 and vl <20000 and tx > 0 and tx < 1 and mes >= 1 and mes <=20:
    simples = vl * tx * mes
    mont = (vl*((1+tx)**mes))
    juComp = mont - vl
    totSim = vl + simples
    print('DIFERENCA DE VALOR = {}'.format(mont-totSim))
    print('JUROS SIMPLES = {}'.format(simples))
    print('JUROS COMPOSTO = {}'.format(juComp))

    


#QUESTAO 3
ver = int(1)

while ver == 1:
    a = float(input())

    b = a.as_integer_ratio()

    if a > 0 and a == b[0]:
        c = float(input())
        d = c.as_integer_ratio()

        if c > 0 and c == d[0]:
            print("media = {}".format((a+c)/2)) 
            print('novo calculo (1-sim 2-nao)')
            ver = int(input())
    else:
        print("nota invalida")
        ver = 0
    


#QUESTÃO 4
a = input().split()

hi = a[0]

mi = a[1]

hf = a[2]

mf = a[3]

if int(mf) > 1 and int(hf) < 24:
    ht = int(hf) - int(hi)
    mt = int(mf) - int(mi)
    print("O JOGO DUROU {} HORA(S) E {} MINUTO(S)".format(ht,mt))