from re import L


def is_prime(n): #determino se um numero é primo
    ans=True
    a=int(n**(1/2))
    for i in range(2,a+1):
        if n%i==0:
            ans=False
            break
    return ans

list_primes=[2]
i=3
while i <1000000:
    if is_prime(i):
        list_primes.append(i)
    i+=1
l=len(list_primes)
print(l) #faço a lista de primos
print(max(list_primes))

for j in range(2,l): #vejo quantas parcelas pode ter no maximo
    if sum(list_primes[0:j])> max(list_primes):
        a=j-1
        print(f'No máximo temos uma soma de {j-1} parcelas.')
        break
ans=True
#descubro um primo que tem o maior numero possivel de parcelas na soma
for j in range(a,2,-1):
    if ans:
        for k in range(0,l-j):
            sum_jprimes=sum(list_primes[k:j+k])
            if sum_jprimes>max(list_primes):
                break
            elif sum_jprimes in list_primes:
                n=j
                m=k
                print(sum_jprimes,n,m)
                ans=False
                break
    else:
        break
for i in range(1,l-n):
    sum_jprimes=sum(list_primes[i:n+i])
    if sum_jprimes in list_primes:
        m=i
        print(sum_jprimes,n,m)