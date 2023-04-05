# Logistic Regression

#I just want to solve a classification problem

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# I am going to write all the functions that I need, an then I will use them in proper order.

#1st function: get the data in three vector x and y. 
#I know before hand how many parameters we will consider, we could assume that we did not know that information

def get_data(file):
    myfile= open(file,'r')    
    data_1=myfile.readlines() #with this I have have every line of the file
    x_1=[]
    x_2=[]
    y=[]
    c=data_1[0].split(',') #it has the titles of each column
    for i in range(1,len(data_1)):
        x_1.append(float(data_1[i].split(',')[0]))
        x_2.append(float(data_1[i].split(',')[1]))
        y.append(float(data_1[i].split(',')[2]))
    myfile.close()
    return ([c,x_1,x_2,y])

#caption=get_data('table3.txt')[0]
#x_1=get_data('table3.txt')[1]
#x_2=get_data('table3.txt')[2]
#y=get_data('table3.txt')[3]
#X=[x_1,x_2]#this way  I garantee that after the second row all variables are int.
#print(caption)
#print(x_1)
#print(x_2)
#print(y)
#print(X)

#feature scalling

def a_scalling(x):
    mean=np.mean(x)
    dif=max(x)-min(x)
    x=(x-mean)/dif
    return x

#x_1=a_scalling(x_1)
#x_2=a_scalling(x_2)
#y=a_scalling(y)

#print(x_1, x_2, y)

def model(X,w,b): #w is a vector, x can be a matrix, we need to transpose so that matrix multiplication makes sense. #returns a vector with th
    z=np.matmul(np.transpose(X),np.transpose(w))+b
    z=np.exp(-1*z)
    z=1/(1+z)
    return z
w=[1,-0.2]
b=2
#print(model(X,w,b))

#we cannot use the cost function of linear regression because for this model, it won't be a convex function.
#We will work it out by defining  loss function an then we will use it in the cost function
# the loss fuction for a certain x_1,x_2,w,b,y is: -y*log(model)-(1-y)*log(1-model)
def logistic_loss_function(X,y,w,b):
    f1=model(X,w,b) #vector will the predicted values
    ones=np.ones(len(f1))
    f0=ones-f1 # vector of 1-predicted values
    a=-np.log(f1)
    b=-np.log(f0)
    p1=np.matmul(y,np.transpose(a)) #actual values * log of predicted values
    p2=np.matmul((ones-y),np.transpose(b)) #(1-actual values) * log of( 1- predicted values)
    return p1+p2

def cost_function (X,y,w,b): #X is the matrices with all the data
    n=len(y)
    cost=logistic_loss_function(X,y,w,b)/n
    return cost

#print(cost_function(X,y,w,b))

def derivatives(X,y,w,b):
    y_m=model(X,w,b)
    n=len(y)
    dif=(y_m-y)
    d_w=np.matmul(dif,np.transpose(X))
    d_w=d_w/n
    d_b=sum(dif)/n
    return [d_w,d_b]
#D=derivatives(X,y,w,b)
#print(D)
#dw=D[0]
#db=D[1]

#alpha=0.001
def iteration(w,dw,b,db,alpha):
    #print(w)
    #print(b)
    w=w-alpha*dw
    b=b-alpha*db
    #print(w)
    #print(b)
    return [w,b]


# Application to an example

#get the data
caption=get_data('table3.txt')[0]
x_1=get_data('table3.txt')[1]
x_2=get_data('table3.txt')[2]
y=get_data('table3.txt')[3]
X=[x_1,x_2]
#print(X)
#print(y)



#feature scalling: it allows for a much bigger alpha but it is easy to lose sense of the values

#x_1=a_scalling(x_1)
#x_2=a_scalling(x_2)
#y=a_scalling(y)
#X=[x_1,x_2]
#print(X)
#print(y)


#plot the original data
colors=[]
for i in range(len(y)):
    if y[i]==0:
        colors.append('blue')
    else:
        colors.append('red')
#plt.scatter(x_1,x_2,c=colors)
#plt.show()

#set parameters
w=[0,0]
b=0

c_v=[cost_function(X,y,w,b)] #vector with the cost for each iteration
l_v=[[w,b]] # vector with the value of the parameters for each iteration

for i in range(10000+1): #doing ?+1 iterations
    dw=derivatives(X,y,w,b)[0]
    db=derivatives(X,y,w,b)[1]
    a=iteration(w,dw,b,db,alpha=0.01)
    w=a[0]
    b=a[1]
    l_v.append(a)
    c_v.append(cost_function(X,y,w,b)) #to check if the cost is decreasing
#print(c_v)
#print(len(c_v))
#print(l_v)
#print(len(l_v))

print(f'The linear equation that minimizes the cost is: y={w[0]}x_1+{w[1]}x_2+{b}')

#plot the cost vs iterations

#x_cord=np.linspace(0,len(c_v),len(c_v))
#plt.plot(x_cord, c_v)
#plt.show()


#plot our approximation in the graph


def aux_function(x1,w0,w1,b,boundary): #this function takes the parametes and draws a line 
    #just write the model in order to the age
    #assuming w1 not zero
    #boundary=probability that it is malignant
    x2=(np.log((1-boundary)/(boundary))-w0*x1-b)/w1
    return x2
print(w[0])
print(w[1])

plt.scatter(x_1,x_2,c=colors)
line_2=np.linspace(0,4,50)
plt.plot(line_2, aux_function(line_2,w[0],w[1],b,0.01))
plt.plot(line_2, aux_function(line_2,w[0],w[1],b,0.7))
plt.plot(line_2, aux_function(line_2,w[0],w[1],b,0.9))
plt.legend(['', '0.01 threshold','0.7 threshold','0.9 threshold'], loc='upper left')
plt.show()

w=[1,-0.2]
b=1
X=[1,15]
print(X)
print(model(X,w,b))