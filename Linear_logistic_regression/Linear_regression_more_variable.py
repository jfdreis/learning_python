#I write an algorithm for the gradient descent, for more then one variable.
#I will do feature scalling. Many things are going to be similar to the case of one parameter
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# I am going to write all the function that I need, an then I will use them in proper order.

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
        x_1.append(int(data_1[i].split(',')[0]))
        x_2.append(int(data_1[i].split(',')[1]))
        y.append(int(data_1[i].split(',')[2]))
    myfile.close()
    return ([c,x_1,x_2,y])

#caption=get_data('table2.txt')[0]
#x_1=get_data('table2.txt')[1]
#x_2=get_data('table2.txt')[2]
#y=get_data('table2.txt')[3]
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

def model(X,w,b): #w is a vector, x can be a matrix, we need to transpose so that matrix multiplication makes sense
    return np.matmul(np.transpose(X),np.transpose(w))+b

#w=[1,2]
#b=2
#print(model(X,w,b))

def cost_function (X,y,w,b): #X is the matrices with all the data
    y_m=model(X,w,b) #one predict the prices
    n=len(y)
    dif=(y_m-y)
    cost=np.dot(dif,dif)
    cost=cost/(2*n)
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
caption=get_data('table2.txt')[0]
x_1=get_data('table2.txt')[1]
x_2=get_data('table2.txt')[2]
y=get_data('table2.txt')[3]
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

#ax = plt.axes(projection='3d')
#ax.scatter3D(x_1, x_2, y)
#ax.set_xlabel('House Area in m^2')
#ax.set_ylabel('# bedrooms')
#ax.set_zlabel('Price times in k euros')
#plt.show()

#set parameters
w=[0,0]
b=0

c_v=[cost_function(X,y,w,b)] #vector with the cost for each iteration
l_v=[[w,b]] # vector with the value of the parameters for each iteration

for i in range(100+1): #doing ?+1 iterations
    dw=derivatives(X,y,w,b)[0]
    db=derivatives(X,y,w,b)[1]
    a=iteration(w,dw,b,db,alpha=0.000001)
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

x_cord=np.linspace(0,len(c_v),len(c_v))
plt.plot(x_cord, c_v)
plt.show()



#Using linear_model.LinearRegression()

regr = linear_model.LinearRegression()
regr.fit(np.transpose(X), y)

predictedprice = regr.predict([[120, 2]])
print(predictedprice)
#print(model([120,2],w,b))

#plot my model and linear_model.LinearRegression()

"""
xline = np.linspace(0, 500, 1000) #coordinates for the area
yline = np.linspace(0, 5, 1000) #coordinates for the number of bedrooms
ax.set_xlabel('House Area in m^2')
ax.set_ylabel('# bedrooms')
ax.set_zlabel('Price times in k euros')

zline=[]
zline_2=[]
for i in range(len(xline)):
    zline.append(model([xline[i],yline[i]],w,b))
    zline_2.append(float(regr.predict([[xline[i],yline[i]]])))
ax.plot3D(xline, yline, zline, 'gray')
ax.plot3D(xline, yline, zline_2, 'blue')

#print(zline_2)
print(len(xline),len(zline_2))
plt.show()
"""

###############################################


