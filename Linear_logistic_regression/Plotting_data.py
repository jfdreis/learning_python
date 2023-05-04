#I want to plot some data

#ploting data related to linear regression
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

from mpl_toolkits import mplot3d # to plot 3D diagrams

##############################################3

#Linear Regression one parameter

################################################3

#I will stat with table 1, house area vs house prices

"""
def get_data(file):
    myfile= open(file,'r')    
    data_1=myfile.readlines() #with this I have have every line of the file
    x=[]
    y=[]
    for i in range(0,len(data_1)):
        x.append(float(data_1[i].split(',')[0]))
        y.append(float(data_1[i].split(',')[1]))
    myfile.close()
    return ([x,y])

x=get_data('table1.txt')[0]
y=get_data('table1.txt')[1]
#print(x)
#print(y)

model_linear=np.poly1d(np.polyfit(x, y, 1))
model_quadratic=np.poly1d(np.polyfit(x, y, 2))
line=np.linspace(0,int(max(x)),int(max(x))+1)

plt.plot(line,model_linear(line))
plt.plot(line,model_quadratic(line))
plt.scatter(x, y)
plt.legend(['linear approximation', 'quadratic'], loc='upper left')
plt.show()

x=75
print(f'The price prediction for a house of {750}m^2 is: \n {model_linear(75)}k euros for the  linear model \n {model_quadratic(75)}k euros for the  quadratic model')

"""
#############################################################

#Linear Regression 2 Parameters

##########################################################

#Here I plot data using table 2. I want a 3D diagram house area  number of bedrooms vs house price

############

#Get data

###############

"""
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


caption=get_data('table2.txt')[0]
x_1=get_data('table2.txt')[1]
x_2=get_data('table2.txt')[2]
y=get_data('table2.txt')[3]
X=[x_1,x_2]#this way  I garantee that after the second row all variables are int.
"""

###################

#Use linear regression on datta collected

#######################
"""
regr = linear_model.LinearRegression()
model=regr.fit(np.transpose(X), y)
z_pred = model.predict(np.transpose(X))
pred=model.predict([[120,2]])
print(pred)

ax=plt.axes(projection='3d')
ax.scatter3D(x_1,x_2,y)
ax.set_xlabel('Area in m^2')
ax.set_ylabel('Number of Bedrooms')
ax.set_zlabel('Price')
"""
#################
# 
#  To plot a surface we will now define a grid


###################

"""
N=75
x_values=np.linspace(0,500,N)
y_values=np.linspace(0,10,N)

x_g, y_g=np.meshgrid(x_values,y_values) # define a grid

"""


#plt.scatter(x_g,y_g,s=0.75)
# I am not being able to write something like: z=model.predict(x_g,y_g)
#Hence I have to turn this problem around. I will use scatter3D
#we now define the z_values

"""
X=[]
for i in range(len(x_g)):
    for j in range(len(x_g)):
        X.append([x_g[i][j],y_g[i][j]])
z=model.predict(X)
x_cord=[]
for i in range(len(x_g)):
    for j in range(len(x_g)):
        x_cord.append(x_g[i][j])
y_cord=[]
for i in range(len(x_g)):
    for j in range(len(x_g)):
        y_cord.append(y_g[i][j])
print(z)
print(len(z))
print(len(x_cord))
ax=plt.axes(projection='3d')
ax.scatter3D(x_1,x_2,y)
ax.scatter3D(x_cord,y_cord,z)

plt.show()
"""


##################################

# I was able to use ax.plot_surface(X,Y,Z) and ax.plot_wireframe(X,Y,Z)
#####################

#get the model coeficients
"""
a=model.coef_
b=model.intercept_
#print(a)
#print(b)

def function_z(x,y,a,b):
    return a[0]*x+a[1]*y+b
z=function_z(x_g,y_g,a,b)
ax=plt.axes(projection='3d')
ax.scatter3D(x_1,x_2,y)
#ax.plot_surface(x_g,y_g,z)
ax.plot_wireframe(x_g,y_g,z)


plt.show()
"""


###########################################

# Logistic regression


##############################

#################

#get data

##################


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
        y.append(int(data_1[i].split(',')[2]))
    myfile.close()
    return ([c,x_1,x_2,y])

caption=get_data('table3.txt')[0]
x_1=get_data('table3.txt')[1]
x_2=get_data('table3.txt')[2]
y=get_data('table3.txt')[3]
X=[x_1,x_2]#this way  I garantee that after the second row all variables are float.
#print(caption)
#print(x_1)
#print(x_2)
#print(y)
#print(X)

#############

#plot the data in 2D

###########

colors=[]
for i in range(len(y)):
    if y[i]==0:
        colors.append('blue')
    else:
        colors.append('red')
plt.scatter(x_1,x_2,c=colors)
plt.show()


#### ######
# 
# plot in 3D

######
"""
ax=plt.axes(projection='3d')
ax.scatter3D(x_1,x_2,y)
plt.show()
"""
#############

##Use logistic regression on the data

###############3

#Want to work with X transpose
X=np.transpose(X)
model = LogisticRegression(solver='liblinear', random_state=0)
model.fit(X,y)
#print(model.classes_)
a=model.coef_
b=model.intercept_
#print(f'the coefficiets are {a}')
#print(f'the interception is at {b}')
#print(model.predict_proba(X))
#print(model.predict(X))


################3

# Plot the predicition

#We are going to plot the wrong predictions, the other "right predictions" are already in the diagram

###########

y_pred=model.predict(X)
"""
x_1_wrong=[]
x_2_wrong=[]
y_wrong=[]
#print(y_pred[0])
#print(y[0])
for i in range(len(y)):
    if y_pred[i]!=y[i]:
        x_1_wrong.append(x_1[i])
        x_2_wrong.append(x_2[i])
        y_wrong.append(y_pred[i])   

ax.scatter3D(x_1_wrong,x_2_wrong,y_wrong, marker="x",c="red")
plt.legend(['actual responses', 'wrong prediction'], loc='upper left')

plt.show()

"""
#########################

# I  want to plot the logistic regression

#######################

def function_z(x,y,a,b):
    z=a[0][0]*x+a[0][1]*y+b
    return 1/(1+np.exp(-z))

def prob(x,y,p):
    return 0*x+0*y+p
#Define a grid
"""
N=20
x_values=np.linspace(0,4,N)
y_values=np.linspace(0,50,N)

x_g, y_g=np.meshgrid(x_values,y_values) # define a grid

z=function_z(x_g,y_g,a,b)
ax=plt.axes(projection='3d')
ax.scatter3D(x_1,x_2,y)
#ax.plot_surface(x_g,y_g,z,color="green")
ax.plot_wireframe(x_g,y_g,z,color="green")
"""
############

#I want to plot a line where the  logistic regression takes values p, where p is the probability of having a malignant tumor

#Playing with diagrams

####################333
def aux_function(y,a,z): ## we get the size of the cancer in function of the probability of being malignant and the age of the pacient
    return-(np.log((1-z)/z)+b+a[0][1]*y)/a[0][0]

#Probility of a malignant tumor

"""
n=10
for i in range(n-1):
    p=(i+1)/n
    z_p=prob(x_g,y_g,p)
    x_1_p=aux_function(y_g,a,z_p)
    ax.plot_wireframe(x_1_p,y_g,z_p)
"""
#plt.show()



