#wrote this to find the m^th prime number

#this function determines whether a number is prime
def is_prime(n):
    ans=True
    a=int(n**(1/2))
    for i in range(2,a+1):
        if n%i==0:
            ans=False
            break
    return ans

i=1
a=2
m=10001
while i <=m:
    if is_prime(a):
        i=i+1
    a=a+1
print(a-1)

#print(is_prime(104743))