import timeit

def test1() :
    mult_3_5={3,5}

    for i in range(2,334):
        a={3*i}
        mult_3_5=mult_3_5.union(a)
    for i in range(2,200):
        a={5*i}
        mult_3_5=mult_3_5.union(a)
    ans=sum(mult_3_5)

    print(f'The sum of all multiples of 3 or 5 up to 1000 is {ans}.')

def test2(): # esta é a função mais rápida
    mults=[]
    for i in range (1,1000):
        if i%3==0 or i%5==0:
            mults.append(i)

    print(f'The sum of all multiples of 3 or 5 up to 1000 is {sum(mults)}')


t1=timeit.timeit('test1()', globals=globals(), number=10)
t2=timeit.timeit('test2()', globals=globals(), number=10)
print(t1)
print(t2)

fibo=[1,2,3]
print(fibo)
fibo.append(5)
print(fibo)
