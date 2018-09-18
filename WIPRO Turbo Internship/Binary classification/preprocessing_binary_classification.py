import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig

'''
Pre-processing steps (in sequence) :
-Normalisation
-Baseline correction/detrending : Tries to remove the effects of a linear(default)/constant offset from the data, thus re-adjusting the baseline of the data
-Savitzky-Golay smoothing : Tries to fit a polynomial of given order over a set of observations that come under the window at a particular moment. Filter width = one-fourth the number of samples, order of local polynomial = 2. These are the parameters of the smoothing function and these values are subject to change due to trial and error. Larger the filter-width and smaller the polynomial order, stronger the smoothing
'''

files = ['ExpiredDigene1.csv','ExpiredDigene2.csv','ExpiredDigene3.csv','ExpiredDigene4.csv','NewDigene1.csv','NewDigene2.csv','NewDigene3.csv','NewDigene4.csv','NewDigene5.csv','NewDigene6.csv','NewDigene7.csv','NewDigene8.csv','NewDigene9.csv','NewDigene10.csv']
path = "/home/WIPRO Turbo Internship/Datasets/Digene/DigeneData/"

fig,ax = plt.subplots()

for i in range(0,len(files)) :
	#Reading the inputs from a file where first column is wavelength and second column is intensity...
	f_name = path + files[i]
	wavelength,intensity = [],[]
	with open(f_name, "r") as filestream :
		for line in filestream :
			currentline = line.split(",")
			wavelength.append(float(currentline[0]))
		    	intensity.append(float(currentline[1]))

	null_indices = []
	null_indices = np.argwhere(np.isnan(intensity))
	
	if (len(null_indices) > 0) :
		wavelength = [v for k,v in enumerate(wavelength) if k not in null_indices]
		intensity = [v for k,v in enumerate(intensity) if k not in null_indices] 
#	if (i == 4) :
#		print intensity
#		print np.mean(intensity)
#		print np.std(intensity)
	
	#Normalisation
	norm = np.copy(intensity)
	norm -= np.mean(norm)
	norm /= np.std(norm)

	#Baseline correction/detrending...
#	if (i == 4) :
#		print norm
	base_corr = sig.detrend(norm,type = 'linear')

	#print base_corr - intensity

	#Savitzky-Golay smoothing (low-pass filter)...
	smooth = sig.savgol_filter(base_corr,57,2)

	#Plotting the original data...
#	ax.plot(wavelength,intensity,'-r',label = 'Unnormalised')

	#Plotting the normalised data...
#	ax.plot(wavelength,norm,'-b',label = 'Normalised')

	#Plotting the baseline corrected data...
#	ax.plot(wavelength,base_corr,'-y',label = 'Baseline corrected')

	#Plotting the smoothened data...
	print i
	name = files[i][:-4]
	if (i != 4) :
		ax.plot(wavelength,smooth,label = name)

#Axis labelling  and legend
plt.xlabel('Wavelengths')
plt.ylabel('Intensities')
#legend = ax.legend(loc = 'upper left', shadow = True)
plt.savefig('preprocessing_without_new_digene1.png')
#plt.show()
