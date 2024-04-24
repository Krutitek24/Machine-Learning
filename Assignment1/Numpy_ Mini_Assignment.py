""" This file is for Assignment#1 Numpy Mini-Assignment.In this assignment, I Get the Banknote Data File and then Read & Split the Data. After doing that will Summarize and Graph It.

Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024"""



import numpy as np
from matplotlib import pyplot as plt

##Read It

data= np.genfromtxt("banknote_data.csv", delimiter=",",skip_header=1,dtype=float)# Read data from the file.

headers=(np.genfromtxt("banknote_data.csv", delimiter=",",names=True)).dtype.names#The first array that holds the string headers from the first row.
print("\n\n")
print("The first array that holds the string headers from the first row:")

print(headers)
second_array=data[ :,:4]#array that holds the numeric data from the first 4 columns.
print("\n\n")
print("Second array that holds the numeric data from the first 4 columns:")
print(second_array)
print("\n\n")
final_array=data[:,4].astype(int)#The final array that holds the numeric data from the final column using type int.
print("The final array that holds the numeric data from the final column using type int:")
print(final_array)
print("\n\n")

##Summarize It,
print("\t\t\t\t\t\t\t\t\tSummary")

print("        -----------------------------------------------------------------")

print("       \t|\t %s\t|\t%s\t|\t%s\t|\t%s"%(headers[0],headers[1],headers[2],headers[3]))
print("        -----------------------------------------------------------------")
print("Minimum\t|\t %f\t|\t%f\t|\t%f\t|\t%f   |"%(second_array[:,0].min(),second_array[:,1].min(),second_array[:,2].min(),second_array[:,3].min()))
print("Maximum\t|\t %f\t|\t%f\t|\t%f\t|\t%f    |"%(second_array[:,0].max(),second_array[:,1].max(),second_array[:,2].max(),second_array[:,3].max()))
print("Mean   \t|\t %f\t|\t%f\t|\t%f\t|\t%f   |"%(second_array[:,0].mean(),second_array[:,1].mean(),second_array[:,2].mean(),second_array[:,3].mean()))
print("Median \t|\t %f\t|\t%f\t|\t%f\t|\t%f   |"%(np.median(second_array[:,0]),np.median(second_array[:,1]),np.median(second_array[:,2]),np.median(second_array[:,3])))
print("\n\n")

##Graph It

plt.figure(1)#Scatter plot
plt.title("2D scatter plot")

plt.xlabel(headers[0])#labels for x-axies.
plt.ylabel(headers[3])#labels for y-axies.
plt.xlim(xmin=second_array[:,0].min()-1, xmax=second_array[:,0].max()+1)#setting the x-axis limits.
plt.ylim(ymin=second_array[:,3].min()-1, ymax=second_array[:,3].max()+1)#setting the y-axis limits.

plt.scatter(second_array[:,0][final_array==1], second_array[:,3][final_array==1], c="aqua", marker="2")# markers for category 1
plt.scatter(second_array[:,0][final_array==0], second_array[:,3][final_array==0], c="yellow", marker=".")#markers for category 0
plt.show()# Show the plot
plt.figure(2)#Bar graph
mean=np.array([second_array[:,0].mean(),second_array[:,1].mean(),second_array[:,2].mean(),second_array[:,3].mean()]) # One dimention array with e mean value for each of the first four columns
plt.bar(headers[:4],mean,color='teal')
plt.ylabel('Mean')#label for y-axies
plt.title('Bar Graph ')

plt.show()# Show the plot