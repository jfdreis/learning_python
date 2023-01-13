
from operator import length_hint


fibo=[1,2,3]
print(fibo)

i=3
d=0
while d <= 4000000:
    c=fibo[i-1]+fibo[i-2]
    d=c+fibo[i-1]
    fibo.append(c)
    i=i+1
print(fibo)

even_fibo=[]
for i in range(0,len(fibo)):
    if fibo[i] % 2==0:
        even_fibo.append(fibo[i])
print(even_fibo)
print(sum(even_fibo))