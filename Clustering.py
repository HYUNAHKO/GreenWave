import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os

def apply_kmeans_and_save_heatmap(image, save_path):
    # Load the image
    image = np.array(image)

    # Extract only the RGB channels (ignore the alpha channel if present)
    rgb_image = image[:, :, :3]

    # Reshape the image to a 2D array of pixels (each row is a pixel, each column is a color channel)
    rgb_pixels = rgb_image.reshape(-1, 3)

    # Perform K-means clustering on the pixel data
    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(rgb_pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_]

    # Reshape the clustered data back to the original image shape
    rgb_clustered_image = clustered.reshape(rgb_image.shape).astype(np.uint8)

    # Identify the cluster corresponding to the green color (assuming this is the algal bloom)
    cluster_centers = kmeans.cluster_centers_
    darkest_cluster_index = np.argmin(np.sum(cluster_centers, axis=1))

    # Create a mask where the darkest cluster (presumed algal bloom) is set to True
    algal_bloom_mask = (kmeans.labels_.reshape(rgb_image.shape[:2]) == darkest_cluster_index)

    # Initialize an intensity image with zeros
    algal_bloom_intensity = np.zeros_like(kmeans.labels_, dtype=float)

    # Calculate the distance of each pixel in the darkest cluster from the cluster center
    distances_to_darkest = np.linalg.norm(rgb_pixels - cluster_centers[darkest_cluster_index], axis=1)
    algal_bloom_intensity[kmeans.labels_ == darkest_cluster_index] = distances_to_darkest[kmeans.labels_ == darkest_cluster_index]

    # Reshape the intensity data back to the original image shape
    algal_bloom_intensity_image = algal_bloom_intensity.reshape(rgb_image.shape[:2])

    # Apply a logarithmic function to the intensity values to emphasize strong areas and de-emphasize weak areas
    adjusted_intensity_log = np.log1p(algal_bloom_intensity_image)

    # Ensure the saved image has the same size as the original image
    plt.figure(figsize=(rgb_image.shape[1] / 100, rgb_image.shape[0] / 100))  # Adjust figsize to match image dimensions
    plt.imshow(adjusted_intensity_log, cmap='viridis')
    plt.title("Log-Adjusted Algal Bloom Intensity Heatmap")
    plt.colorbar(label='Intensity (Log-Adjusted)')
    
    # Save the heatmap
    plt.savefig(save_path)  # Save as B.png or the provided save_path
    plt.close()

    # Debugging information
    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"Heatmap image was not saved: {save_path}")

    return save_path
