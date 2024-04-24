"""
Image Quantization using K-Means Clustering for Assignment 5
------------------------------------------------------------

This script demonstrates image quantization using the k-means clustering algorithm.
It includes functions to quantize an image with different numbers of clusters,
display the resulting images, and plot the inertia (SSE) for each quantization.
It also shows how to reuse a k-means model to quantize a completely different image.


|================================================================================================================|
|For the above data, there is no obvious “elbow”. Maybe there is a little bit of an elbow at k=4 , k=6 or k=8    |
|Because there was no obvious elbow, I used k=14 for Task 2, which is the quantization I liked best.             |
|================================================================================================================|




Kruti Patel(000857563),Machine Learning - COMP-10200, Mohawk College, Winter-2024

"""
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt
from skimage import io

def quantize_image(image, k):
    
    """
    The quantize_image function takes an image path and a number of clusters as input,
    and returns the quantized image and the k-means model.
    
    
    :param image: Specify the path to the image that will be quantized
    :param k: Determine the number of clusters to use in k-means clustering
    :return: A quantized image and the k-means model
    :
    """
    # Load the image
    image = io.imread(image)
    original_shape = image.shape  # Save the original shape

    # Reshape the image to a 2D array of pixels
    colors = image.reshape(original_shape[0] * original_shape[1], 3)

    # Apply k-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=5)
    kmeans.fit(colors)

    # Create a new image map by replacing each pixel with its closest centroid
    new_colors = kmeans.cluster_centers_[kmeans.labels_]
    quantized_image = new_colors.reshape(original_shape).astype('uint8')

    return quantized_image, kmeans

def quantize_different_image(model, image):
    
    """
    The quantize_different_image function takes a KMeans model and an image path as input.
    It loads the image, reshapes it to a 2D array of pixels, predicts cluster assignments for each pixel using the model, 
    replaces each pixel with its closest centroid (using the labels), and returns this new quantized image.
    
    :param model: Pass in the kmeans model
    :param image: Specify the path to the image that will be quantized
    :return: A quantized image
    :
    """
    # Load the image
    image = io.imread(image)
    original_shape = image.shape  # Save the original shape

    # Reshape the image to a 2D array of pixels
    colors = image.reshape(original_shape[0] * original_shape[1], 3)

    # Predict cluster assignments for each pixel
    labels = model.predict(colors)

    # Replace each pixel with its closest centroid
    new_colors = model.cluster_centers_[labels]
    quantized_other_image = new_colors.reshape(original_shape).astype('uint8')

    return quantized_other_image


ks = list(range(2,20,2))# list of number of clusters from 2 to 18 with step size 2
inertias = []# List to store the Inertia values corresponding to different numbers of clusters.
def variety_k():   
    """
    The variety_k function takes no arguments and returns nothing.
    It simply iterates through a list of k values, quantizes the image with each value of k, 
    and displays the resulting images along with their SSEs.
    
    :return: Nothing
    :
    """
    for k in ks:# Quantize the first image for different value of k
        quantized_image,kmeans_model = quantize_image("nature.jpg", k=k)
        inertias.append(kmeans_model.inertia_)#store the SSE for each images
        print("k =",k,",  SSE =",kmeans_model.inertia_)# Displaying the results
        plt.figure()# Create a new figure window
        plt.imshow(quantized_image)# Showing the resultant image
        plt.title(' nature.jpg    K=%d' % k)
        plt.show()
plt.figure()
plt.imshow(io.imread("nature.jpg"))# show the original image
plt.title('nature.jpg')
plt.axis('off')
plt.show()       
variety_k()   
# Plotting the graph 
plt.title("nature.jpg   inertia vs. k")
plt.xlabel("k")
plt.ylabel("inertia")
plt.plot(ks, inertias)
plt.show()
# Assume the k-means model from the first quantization is to be reused
# Quantize a completely different image
quantized_image,kmeans_model = quantize_image("nature.jpg", k=14)
different_image = "cute.jpg"  # Update this path
plt.figure()
plt.imshow(io.imread("cute.jpg"))# show the new image
plt.title('cute.jpg')
plt.axis('off')
plt.show()
#Assume the k-means model from the first quantization is to be reused to quantize a completely different image
quantized_different_image = quantize_different_image(kmeans_model, different_image)
plt.figure()
plt.imshow(quantized_different_image)# display the quantized new image
plt.title(' cute.jpg  Quantized Using Image  natura1.jpg')
plt.axis('off')
plt.show()
