#I write an algorithm for the gradient descent, I test it with a very small table
import matplotlib.pyplot as plt
import numpy as np
#from sklearn import linear_model


# I am going to write all the functions that I need.

#1st function: get the data in two vector x and y. 

def get_data(file):
    myfile= open(file,'r')    
    data_1=myfile.readlines() #with this I have have every line of the file
    x=[]
    y=[]
    for i in range(0,len(data_1)):
        x.append(int(data_1[i].split(',')[0]))
        y.append(int(data_1[i].split(',')[1]))
    myfile.close()
    return ([x,y])
    
#x=get_data('table1.txt')[0]
#y=get_data('table1.txt')[1]
#print(x)
#print(y)

#2nd function we write a cost function
def cost_function (w,b,x,y):
    m=len(x)
    cost=0
    for i in range(m):
        cost = cost + (w*x[i]+b-y[i])**2
    cost=cost/(2*m)
    return cost

#3rd function: we compute the value of the derivatives in order to use  the gradient descent algortihm
def derivatives(w,b,x,y):
    derivative_w=0
    derivative_b=0
    for i in range(len(x)):
        derivative_w=derivative_w+(w*x[i]+b-y[i])*x[i]
        derivative_b=derivative_b+(w*x[i]+b-y[i])
    #    print(derivative_w)
    derivative_w=derivative_w/len(x)
    #print(derivative_w)
    derivative_b=derivative_b/len(x)
    return [derivative_w,derivative_b]

#4th function: onve we have the derivatives we can actualize the values of the parameters w and b
def iteration(w,dw,b,db,alpha):
    #print(w)
    #print(b)
    w=w-alpha*dw
    b=b-alpha*db
    #print(w)
    #print(b)
    return [w,b]

def my_model(x,w,b):
    return w*x+b
#Example of application

#1st. I want to plot the inital data
#2nd. I Want to plot a diagram where one sees whether the cost function is decreasing
#3rd. I want to plot a diagram where the linear approximation appear

data=get_data('table1.txt')
x=data[0]
y=data[1]

#plt.scatter(x, y)
#plt.title("House Price vs Area")
#plt.axis([0, 11*max(x)/10, 0, 11*max(y)/10]) #[xmin, xmax, ymin, ymax]
#plt.xticks(np.arange(0,11*max(x)/10,5))
#plt.yticks(np.arange(0,11*max(y)/10,5))
#plt.xlabel("House Area times 10m^2")
#plt.ylabel("Price times 10k euros")
#plt.show()

#set w=0, b=0
w=0
b=0
#vectors to plot the cost function
c_x=[] 
c_y=[] 
#vectors to plot the linear approximation
l_w=[0]
l_b=[0]
for i in range(100+1): #doing ?+1 iterations
    c_x.append(i)
    c_y.append(cost_function(w,b,x,y)) #to check if the cost is decreasing
    dw=derivatives(w,b,x,y)[0]
    db=derivatives(w,b,x,y)[1]
    a=iteration(w,dw,b,db,alpha=0.0025)
    w=a[0]
    b=a[1]
    l_w.append(w)
    l_b.append(b)
#print(w,b)

#plot cost function
#plt.plot(c_x,c_y)#marker='o' to have dots in the data
#plt.title("Cost function")
#plt.axis([0, 11*max(c_x)/10, 0, 11*max(c_y)/10]) #[xmin, xmax, ymin, ymax]
#plt.xlabel("Iteration")
#plt.ylabel("R-squared value")
#plt.show()


#plot the linear approximation
l=[l_w,l_b]
x_cord= np.arange(0,50+1,1)
#print(x_cord)

for i in range((len(c_x))): # I want to run through all linear approximations that we did
    y_cord=[]
    for j in range(len(x_cord)):
        y_cord.append(l[0][i]*x_cord[j]+l[1][i]) #l has the values that we obtained for the parameters w and b
        #plt.plot(x_cord,y_cord) #we are plotting the linear approximation
        #plt.title("House Price vs Area")
        #plt.xticks(np.arange(0,1.05*max(x),5))
        #plt.yticks(np.arange(0,1.05*max(y),10))
        #plt.axis([0, 1.01*max(x), 0,1.01*max(y)])
        #plt.xlabel("House Area times 10m^2")
        #plt.ylabel("Price times 10k euros")
        #plt.axhline(y=0,color='k')
        #plt.axvline(x=0, color='k')

#plt.scatter(x, y)
#plt.show()

print(f'The linear equation that minimizes the cost is: y={w}x+{b}')

#I want to compare what I obtained with the functions already existing in python

#np.polyfit(x, y, 1) gets a polynomial equation to fit the data x, y of degree n
#np.poly1d defines a polynomial according to the coeficients
model_2 = np.poly1d(np.polyfit(x, y, 1))
line_2=np.linspace(0,50,51)

plt.plot(line_2, my_model(line_2,w,b))
plt.plot(line_2,model_2(line_2))
plt.scatter(x, y)
plt.legend(['my_model', 'theirs_model'], loc='upper left')
plt.show()


