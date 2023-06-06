import math

d=10000
dict_1={}
for i in range(1,d+1):
    if i%2==0:
        for j in range(int(2*i/5 -1),int(3*i/7 + 1),2):
            if math.gcd(j,i)==1:
                dict_1[f'{j}/{i}']=j/i
    else:
        for j in range(int(2*i/5 -1),int(3*i/7 + 1)):
            if math.gcd(j,i)==1:
                dict_1[f'{j}/{i}']=j/i


#print(dict_1)


list_1=dict_1.values()
list_1=sorted(list_1)
#print(list_1)
position=list_1.index(3/7)-1 # posição na lista ordenada que se quer.
want=list_1[position]
print(want)

list_2=list(dict_1.values())
a=list_2.index(want)
list_3=list(dict_1.keys())
print(list_3[a])

#aqui sabemos que para d=10k temos 4280/9987. Ora, isso significa que para d=1000000 podemos diminuir muito a busca

d=1000000
dict_1={}
for i in range(1,d+1):
    if i%2==0:
        for j in range(int(4280*i/9987 -1),int(3*i/7 + 1),2):
            if math.gcd(j,i)==1:
                dict_1[f'{j}/{i}']=j/i
    else:
        for j in range(int(4280*i/9987 -1),int(3*i/7 + 1)):
            if math.gcd(j,i)==1:
                dict_1[f'{j}/{i}']=j/i


#print(dict_1)


list_1=dict_1.values()
list_1=sorted(list_1)
#print(list_1)
position=list_1.index(3/7)-1 # posição na lista ordenada que se quer.
want=list_1[position]
print(want)

list_2=list(dict_1.values())
a=list_2.index(want)
list_3=list(dict_1.keys())
print(list_3[a])




