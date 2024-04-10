import numpy as np
import matplotlib.pyplot as plt

# Define the likelihood function
def likelihood(pt, s):
  distance = np.sqrt((pt[0] - s[0])**2 + (pt[1] - s[1])**2)
  return np.exp(-100 * (distance - 0.2)**2)

# Define the 2D space
x = np.linspace(0.0, 1., 100)
y = np.linspace(0.0, 1., 100)
X, Y = np.meshgrid(x, y)

# Define the source location
source_location = [0.3, 0.4]

# Calculate the likelihood for each point in the 2D space
Z = likelihood([X, Y], source_location)

x_r = []
y_r = []
x_g = []
y_g = []
x_i = []
y_i = []
z_i = []
for i in range(100):
  x_1, y_1, u = np.random.uniform(0, 1, size=3)
  if u < likelihood([x_1, y_1], source_location):
      x_g.append(x_1)
      y_g.append(y_1)
      zii = 1.0
  else:
      x_r.append(x_1)
      y_r.append(y_1)
      zii = 0.0
  x_i.append(x_1)
  y_i.append(y_1)
  z_i.append(zii)


# Plot the likelihood function
plt.contourf(X, Y, Z, cmap='gray')
plt.scatter(source_location[0], source_location[1], color='blue', marker='x', label='Source Location')
plt.scatter(x_g, y_g, color='green', alpha=0.3, label='Positive Likelihood')
plt.scatter(x_r, y_r, color='red', alpha=0.3, label='Negative Likelihood')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.title('Problem 1')
plt.show()

# #######2#######

X_s, Y_s = np.meshgrid(x, y)
Z_inv = np.ones_like(X_s)

for i in range(len(x_i)):
  if z_i[i] == 1.0:
      likelihood_point = likelihood([x_i[i], y_i[i]] , [X_s, Y_s] )
  else:
      likelihood_point = 1.0 - likelihood([x_i[i], y_i[i]] , [X_s, Y_s] )
  Z_inv *= likelihood_point

plt.contourf(X_s, Y_s, Z_inv, cmap='gray')
plt.scatter(source_location[0], source_location[1], color='blue', marker='x', label='Source Location')
plt.scatter(x_g, y_g, color='green', alpha=0.3, label='Positive Likelihood')
plt.scatter(x_r, y_r, color='red', alpha=0.3, label='Negative Likelihood')
plt.xlabel('X')
plt.ylabel('Y')
plt.legend(loc='upper right')
plt.title('Problem 2')
plt.show()

######3######

for gr in range(3):
  Z_3  = np.ones([100,100])
  x_3,y_3 = np.random.uniform(0,1,size=2)
  for i in range(len(x_i)):
    u = np.random.uniform(0,1)
    if u > likelihood([x_3, y_3], source_location):
      likelihood_point = likelihood([x_3, y_3] , [X,Y])
    else:
      likelihood_point = 1.0 - likelihood([x_3, y_3] , [X,Y])
    Z_3 *= likelihood_point

  plt.contourf(X, Y, Z_3, cmap='gray')
  plt.scatter(x_3, y_3, color='red', alpha=0.3, label='Negative Likelihood')
  plt.xlabel('X')
  plt.ylabel('Y')
  plt.legend(loc='upper right')
  plt.title('Problem 3')
  plt.show()
  
######4######

# s = np.random.uniform(0,1,size=2)

b_bar = np.ones((100,100))/10000

x_g4 = []
y_g4 = []
x_r4 = []
y_r4 = []
x_i4 = []
y_i4 = []
z_i4 = []
bg_4 = []

x_1, y_1 = np.random.uniform(0, 1, size=2)
for i in range(10):
  u = np.random.uniform(0, 1)
  if u < likelihood([x_1, y_1], source_location):
      x_g4.append(x_1)
      y_g4.append(y_1)
      print("here")
      zii = 1.0
  else:
      x_r4.append(x_1)
      y_r4.append(y_1)
      zii = 0.0
      
  x_i4.append(x_1)
  y_i4.append(y_1)
  z_i4.append(zii)

  if z_i4[i] == 1.0:
      likelihood_point = likelihood([x_i4[i], y_i4[i]] , [X,Y])
  else:
      likelihood_point = 1.0 - likelihood([x_i4[i], y_i4[i]] , [X,Y])
  
  likelihood_point = likelihood_point * b_bar
  b_bar = likelihood_point / np.sum(likelihood_point)
  bg_4.append(b_bar)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
  ax.contourf(X, Y, bg_4[i], cmap='gray')
  ax.scatter(x_i4[:i+1], y_i4[:i+1], color='brown', alpha=0.3, label='Measurements')
  #ax.scatter(x_r4[:i+1], y_r4[:i+1], color='red', alpha=0.3, label='Negative Likelihood')
  #ax.scatter(x_g4[:i+1], y_g4[:i+1], color='green', alpha=0.3, label='Positive Likelihood')
  ax.scatter(source_location[0], source_location[1], color='brown', marker='x', label='Source Location')
  ax.set_title(f'Measurement {i+1}')
# plt.title('Problem 4')
plt.show()


######5######


b_bar = np.ones((100,100))/10000

x_g5 = []
y_g5 = []
x_r5 = []
y_r5 = []
x_i5 = []
y_i5 = []
z_i5 = []
bg_5 = []

for i in range(10):
  x_1, y_1, u = np.random.uniform(0, 1, size=3)
  if u < likelihood([x_1, y_1], source_location):
      x_g5.append(x_1)
      y_g5.append(y_1)
      zii = 1.0
  else:
      x_r5.append(x_1)
      y_r5.append(y_1)
      zii = 0.0
      
  x_i5.append(x_1)
  y_i5.append(y_1)
  z_i5.append(zii)

  if z_i5[i] == 1.0:
      likelihood_point = likelihood([x_i5[i], y_i5[i]] , [X,Y])
  else:
      likelihood_point = 1.0 - likelihood([x_i5[i], y_i5[i]] , [X,Y])
  
  likelihood_point = likelihood_point * b_bar
  b_bar = likelihood_point / np.sum(likelihood_point)
  bg_5.append(b_bar)

fig, axes = plt.subplots(2, 5, figsize=(15, 6))
for i, ax in enumerate(axes.flat):
  ax.contourf(X, Y, bg_5[i], cmap='gray')
  ax.scatter(x_i5[:i+1], y_i5[:i+1], color='brown', alpha=0.3, label='Measurements')
  ax.scatter(source_location[0], source_location[1], color='brown', marker='x', label='Source Location')
  ax.set_title(f'Measurement {i+1}')
# plt.title('Problem 5')
plt.show()
  

