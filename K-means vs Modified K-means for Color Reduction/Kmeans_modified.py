import numpy as np
import math
import random
from PIL import Image
from matplotlib import pyplot as plt

img = Image.open('trial.jpeg', 'r')									#Opening the file
inp = list(img.getdata())											#Pixel values in RGB format
Y = np.array(inp).transpose()										#Pixel values as a (3 x (#pixels)) matrix

m = len(Y)															#No of parameters = 3
N = len(Y[0])														#No of inputs = No of pixels

k = int(raw_input("Enter the number of clusters to be formed : "))							
epsilon = 0.1														#Stopping condition
error = 10000														#Error at each iteration, initialised to a large value
q = 0																#To keep count of number of iterations

init_centroids = np.zeros(shape = (3,k))							#Initial guess for the centroids
centroids = np.zeros(shape = (3,k))									#The centroids for each iteration of the k-means algorithm
err = []															#The error list to be used for plotting

init_centroids[:,0] = Y[:,random.randint(0,N - 1)]					#The first centroid is a randomly picked point from input

#Generating the centroids iteratively...
#In each iteration, a point that is farthest away from its assigned cluster/centroid is chosen to be a new centroid
ite = 1																#Number of iterations in the initialisation step
#Until the required number of centroids are obtained...
while (ite < k) :
	max_sq_dist = -1												#Initialisation											
	count = [0] * ite												#To keep count of the number of points in a particular cluster
	update_cents = np.zeros(shape = (3,k))							#The updated centroids, found before the end of each iteration

	for i in range(0,N) :
		min_sq_dist = 16777217										#1 + (256 ** 3)  ----> Just an initialisation
		#Checking the squared distance of the present data point from each of the presently chosen centroids...
		for j in range(0,ite) :
			sq_dist = 0
			for l in range(0,m) :
				sq_dist += np.square(Y[l][i] - init_centroids[l][j])
			#Assigning the observation point to the cluster to which its centroid is the closest	
			if (sq_dist <= min_sq_dist ) :
				min_sq_dist = sq_dist
				min_cent = j
		count[min_cent] += 1										#Incrementing the count of the number of points in that cluster
		#Update step, part one...
		for j in range(0,m) :
			update_cents[j][min_cent] += Y[j][i]
		
		#The data point whose squared distance from its assigned cluster/centroid is the largest is chosen as a new centroid
		if (max_sq_dist <= min_sq_dist) :
			max_sq_dist = min_sq_dist
			neo_cent = i
	
	#Update step, part two...
	for i in range(0,ite) :
		for j in range(0,m) :
			update_cents[j][i] = int(update_cents[j][i]/count[i])

	init_centroids = update_cents									#Completing the update, to be used for the next iteration

	init_centroids[:,ite] = Y[:,neo_cent]							#Adding the new centroid
	ite = ite + 1
							
centroids = init_centroids											#Using the initial guess for the centroids

while (error > epsilon) :
	count = [0] * k													#Count of number of points belonging to each cluster, initialised to zeros
	
	#Matrix for the centroids to be updated at the end of each iteration
	new_centroids = []
	for i in range (0,m) :
		new_centroids.append(count)
	new_centroids = np.array(new_centroids)
	
	img_inp = []													#List to help generate the images based on clustering
	for i in range (0,N) :											#Iterating over all points
		distances = [0]*k											#Initialising the distances of the centroids from the i-th point to zeros
		for j in range (0,m) :										#Iterating over all parameters of the i-th point
			for l in range (0,k) :									#Iterating over all the centroids
				distances[l] += np.square(centroids[j][l] - Y[j][i])#Squared distances of the i-th point from the l-th centroid
		
		for l in range (0,k) :										#Iterating over all the centroids
			distances[l] = np.sqrt(distances[l])					#The actual distances of the centroids from the i-th point
			
		closest_centroid_index = distances.index(min(distances))	#Index of the closest centroid to the i-th point
		closest_centroid = []										#List that will be converted to a matrix later
																	#This will store the corresponding centroid values for each data point
		#Centroid update and count update
		count[closest_centroid_index] += 1
		for j in range (0,m) :										#Iterating over all parameters
			new_centroids[j][closest_centroid_index] += Y[j][i]
			closest_centroid.append(int(centroids[j][closest_centroid_index]))
		
		img_inp.append(tuple(closest_centroid))						#Pixel values for the image to be generated
		
	#Generating an image from the pixel values associated with the k centroid values...
	name = "Image_mod" + str(q)
	temp = Image.new(img.mode, img.size)
	temp.putdata(list(img_inp))
	temp.save(name,"png")											#Saving the image generated
	
	#Calculating the new centroids for the next iteration...
	for j in range (0,m) :											#Iterating over all parameters
		for l in range (0,k) :										#Iterating over all the centroids
			new_centroids[j][l] = int(new_centroids[j][l]/count[l])
	
	error = 0														#Initialising the error
	#Calculating the error...
	for j in range (0,m) :											#Iterating over all parameters
		for l in range (0,k) :										#Iterating over all the centroids
			error += np.square(centroids[j][l] - new_centroids[j][l])
	error = np.sqrt(error)
	err.append(error)											
	print "Error in iteration " + str(q + 1) + " = ",error
	centroids = new_centroids
	q += 1

#Plotting...
iters = np.linspace(0,q,q)
plt.plot(iters,err,ls = '-',color = 'blue')
plt.xlabel('No of iterations')
plt.ylabel('RMS error')
plt.title('Convergence of modified k-means algorithm')
plt.savefig('convergence_graph.png')
