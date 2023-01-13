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
while i <=10001:
    if is_prime(a):
        i=i+1
    a=a+1
print(a-1)

print(is_prime(104743))