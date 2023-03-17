from re import L
#This Project Euler Problem 50
#The prime 41, can be written as the sum of six consecutive primes:
#41 = 2 + 3 + 5 + 7 + 11 + 13
#This is the longest sum of consecutive primes that adds to a prime below one-hundred.
#The longest sum of consecutive primes below one-thousand that adds to a prime, contains 21 terms, and is equal to 953.
#Which prime, below one-million, can be written as the sum of the most consecutive primes?

def is_prime(n): #determines whether a number is prime
    ans=True
    a=int(n**(1/2))
    for i in range(2,a+1):
        if n%i==0:
            ans=False
            break
    return ans

list_primes=[2]
i=3 #we are already considering the prime number 2
while i <1000000: # we are making a list with all primes up one million
        list_primes.append(i)
        i+=1
l=len(list_primes)
print(l)
print(max(list_primes))

# in the next for we determine an upper bound for the number of consecutive primes one can add
# such that the result is smaller then the greatest prime belowone-million.
# a is that upper bound
# we do this do decrease the number of iterantions in a later for.
for j in range(2,l):
    if sum(list_primes[0:j])> max(list_primes):
        a=j-1
        print(f'At most one can add {j-1} consecutive primes.')
        break
ans=True

#now we try to find the prime below one million that is the sum of as many as possible consecutive primes
#we also save the number of primes that we are adding together
for j in range(a,2,-1):#we go in reverse order
    if ans:
        for k in range(0,l-j):
            sum_jprimes=sum(list_primes[k:j+k]) #sum j consecutives primes
            if sum_jprimes>max(list_primes):
                break
            elif sum_jprimes in list_primes: #if the sum is itself a prime number we "save" all information and get out
                n=j
                m=k
                print(sum_jprimes,n,m)
                ans=False
                break
    else:
        break
#with the previous for we obtain the maximum number of consecutive primes that can be added together to obtain another prime below one million
# it could happen that there we more consecutive primes satisfying these condition. We make sure that we have all those primes in the next for
for i in range(1,l-n):
    sum_jprimes=sum(list_primes[i:n+i])
    if sum_jprimes in list_primes:
        m=i
        print(sum_jprimes,n,m)