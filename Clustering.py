import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from PIL import Image
import os

def apply_kmeans_and_save_heatmap(image, save_path):
    # 이미지 불러오기
    image = np.array(image)

    rgb_image = image[:, :, :3]  # RGB 채널만 추출
    rgb_pixels = rgb_image.reshape(-1, 3)  # 2D 배열로 변환

    kmeans = KMeans(n_clusters=3, random_state=42)
    kmeans.fit(rgb_pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_]

    rgb_clustered_image = clustered.reshape(rgb_image.shape).astype(np.uint8)

    cluster_centers = kmeans.cluster_centers_
    darkest_cluster_index = np.argmin(np.sum(cluster_centers, axis=1))  # 가장 어두운 클러스터 찾기

    algal_bloom_mask = (kmeans.labels_.reshape(rgb_image.shape[:2]) == darkest_cluster_index)

    algal_bloom_intensity = np.zeros_like(kmeans.labels_, dtype=float)

    distances_to_darkest = np.linalg.norm(rgb_pixels - cluster_centers[darkest_cluster_index], axis=1)
    algal_bloom_intensity[kmeans.labels_ == darkest_cluster_index] = distances_to_darkest[kmeans.labels_ == darkest_cluster_index]

    algal_bloom_intensity_image = algal_bloom_intensity.reshape(rgb_image.shape[:2])

    adjusted_intensity_log = np.log1p(algal_bloom_intensity_image)  # 로그 함수 적용

    plt.figure(figsize=(rgb_image.shape[1] / 100, rgb_image.shape[0] / 100))
    plt.imshow(adjusted_intensity_log, cmap='viridis')
    plt.title("Log-Adjusted Algal Bloom Intensity Heatmap")
    plt.colorbar(label='Intensity (Log-Adjusted)')
    
    plt.savefig(save_path)  # 히트맵 저장
    plt.close()

    if not os.path.isfile(save_path):
        raise FileNotFoundError(f"Heatmap image was not saved: {save_path}")

    return save_path
