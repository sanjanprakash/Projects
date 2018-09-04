import numpy as np
import math
import random
from PIL import Image
from matplotlib import pyplot as plt

#input_file = raw_input("Enter the name of the image file with the appropriate extension : ")
img = Image.open('trial.jpeg', 'r')										#Opening the file
inp = list(img.getdata())												#Pixel values in RGB format
Y = np.array(inp).transpose()											#Pixel values as a (3 x (#pixels)) matrix

m = len(Y)																#No of parameters = 3
N = len(Y[0])															#No of inputs = No of pixels
centroids = []
err = []

#The initial guess for the k centroids are k random points from the input matrix Y
k = int(raw_input("Enter the number of clusters to be formed : "))
init_guess = random.sample(range(0, N), k)							
for i in init_guess :
	centroids.append(inp[i])											
centroids = np.array(centroids).transpose()								#Initial guess for the k centroids

#epsilon = float(raw_input("Enter the stopping condition : "))			#Stopping condition
epsilon = 0.1
error = 10000															#Error at each iteration, initialised to a large value
q = 0																	#To keep count of number of iterations

while (error > epsilon) :
	count = [0] * k														#Count of number of points belonging to each cluster, initialised to zeros
	
	#Matrix for the centroids to be updated at the end of each iteration
	new_centroids = []
	for i in range (0,m) :
		new_centroids.append(count)
	new_centroids = np.array(new_centroids)
	
	img_inp = []														#List to help generate the images based on clustering
	for i in range (0,N) :												#Iterating over all points
		distances = [0] * k												#Initialising the distances of the centroids from the i-th point to zeros
		for j in range (0,m) :											#Iterating over all parameters of the i-th point
			for l in range (0,k) :										#Iterating over all the centroids
				distances[l] += np.square(centroids[j][l] - Y[j][i])	#Squared distances of the i-th point from the l-th centroid
		
		for l in range (0,k) :											#Iterating over all the centroids
			distances[l] = np.sqrt(distances[l])						#The actual distances of the centroids from the i-th point
			
		closest_centroid_index = distances.index(min(distances))		#Index of the closest centroid to the i-th point
		closest_centroid = []											#List that will be converted to a matrix later
																		#This will store the corresponding centroid values for each data point
		#Centroid update and count update
		count[closest_centroid_index] += 1
		for j in range (0,m) :											#Iterating over all parameters
			new_centroids[j][closest_centroid_index] += Y[j][i]
			closest_centroid.append(centroids[j][closest_centroid_index])
		
		img_inp.append(tuple(closest_centroid))							#Pixel values for the image to be generated
		
	#Generating an image from the pixel values associated with the k centroid values
	name = "Image" + str(q)
	temp = Image.new(img.mode, img.size)
	temp.putdata(list(img_inp))
	temp.save(name,"png")												#Saving the image generated
	
	#Calculating the new centroids for the next iteration
	for j in range (0,m) :												#Iterating over all parameters
		for l in range (0,k) :											#Iterating over all the centroids
			new_centroids[j][l] = int(new_centroids[j][l]/count[l])
	
	error = 0															#Initialising the error
	#Calculating the error
	for j in range (0,m) :												#Iterating over all parameters
		for l in range (0,k) :											#Iterating over all the centroids
			error += np.square(centroids[j][l] - new_centroids[j][l])
	error = np.sqrt(error)												
	err.append(error)
	print "Error in iteration " + str(q + 1) + " = ",error
	centroids = new_centroids
	q += 1
	
#Plotting...
iters = np.linspace(0,q,q)
plt.plot(iters,err,ls = '-',color = 'red')
plt.xlabel('No of iterations')
plt.ylabel('RMS error')
plt.title('Convergence of k-means algorithm')
plt.savefig('convergence_graph.png')
