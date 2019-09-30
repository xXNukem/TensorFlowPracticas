# -*- coding: utf-8 -*-

tupla=(1,2,3,4,5,6,7) #No puede ser modificada
lista=[2,5,4,3,6,7,4] #Se puede modificar
lista2=[]
print(lista[4]) #Trae el elemento 4 de la lista

lista[4]=14 #Podemos modificar el elemento de la lista
print(lista[4])

del lista[4] #Podemos borrar el elemento 4
print (lista[4])

lista.append(3) #añadimos elementos al final

lista2=lista