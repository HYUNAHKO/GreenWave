import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans
from Clustering import apply_kmeans_and_save_heatmap
import segmentation_models_pytorch as smp
import torch
import torch.nn.functional as F

def apply_kmeans_and_save_heatmap(image, save_path, n_clusters=7):
    """
    이미지에서 K-means 클러스터링을 수행하고 히트맵을 생성하여 저장합니다.
    HSV 색공간에서 클러스터링을 수행하여 녹조를 효과적으로 분류합니다.
    """
    # RGB 채널만 추출합니다 (alpha 채널이 있으면 무시)
    rgb_image = image[:, :, :3]

    # RGB 이미지를 HSV 색 공간으로 변환
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)

    # HSV 이미지를 2차원 픽셀 배열로 재구성
    hsv_pixels = hsv_image.reshape(-1, 3)

    # 픽셀 데이터에 대해 K-means 클러스터링 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(hsv_pixels)
    clustered = kmeans.cluster_centers_[kmeans.labels_]

    # 클러스터링된 데이터를 원본 이미지 형태로 재구성
    hsv_clustered_image = clustered.reshape(hsv_image.shape).astype(np.uint8)

    # 클러스터링된 이미지를 RGB로 다시 변환하여 시각화
    rgb_clustered_image = cv2.cvtColor(hsv_clustered_image, cv2.COLOR_HSV2RGB)

    # 녹색 색상 범위에 가장 가까운 클러스터를 식별
    cluster_centers = kmeans.cluster_centers_
    hue_centers = cluster_centers[:, 0]  # Hue 채널

    # 녹색 색상 범위 (60°-120° in HSV)에 가장 가까운 클러스터를 찾기
    green_cluster_index = np.argmin(np.abs(hue_centers - 90))  # 90은 녹색 범위의 중간값

    # 녹색 클러스터 (녹조로 추정)를 True로 설정하는 마스크 생성
    algal_bloom_mask = (kmeans.labels_.reshape(hsv_image.shape[:2]) == green_cluster_index)

    # 녹조 발생 마스크 이미지를 'F.png'로 저장
    plt.imsave(f"{save_path}/F.png", algal_bloom_mask, cmap='gray')
    print(f"녹조 발생 마스크 이미지가 {save_path}/F.png 에 저장되었습니다.")

    # 강도 이미지 초기화
    algal_bloom_intensity = np.zeros_like(kmeans.labels_, dtype=float)

    # 식별된 클러스터의 각 픽셀에서 클러스터 중심까지의 거리 계산
    distances_to_green = np.linalg.norm(hsv_pixels - cluster_centers[green_cluster_index], axis=1)
    algal_bloom_intensity[kmeans.labels_ == green_cluster_index] = distances_to_green[kmeans.labels_ == green_cluster_index]

    # 강도 데이터를 원본 이미지 형태로 재구성
    algal_bloom_intensity_image = algal_bloom_intensity.reshape(hsv_image.shape[:2])

    # 강도 값에 로그함수 적용하여 강한 영역을 강조하고 약한 영역을 약화
    adjusted_intensity_log = np.log1p(algal_bloom_intensity_image)

    # 강도 이미지와 마스크가 올바르게 적용되었는지 확인
    final_intensity_image = np.zeros_like(adjusted_intensity_log)

    # 녹조 영역에만 강도 이미지 적용
    final_intensity_image[algal_bloom_mask] = adjusted_intensity_log[algal_bloom_mask]

    # 로그 조정된 강도 히트맵을 'B.png'로 저장
    plt.imsave(f"{save_path}/B.png", final_intensity_image, cmap='viridis')
    print(f"히트맵 이미지가 {save_path}/B.png 에 저장되었습니다.")


# 1. 세그멘테이션 모델 로드
def load_segmentation_model(model_path):
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=2)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cuda')))
    model = model.cuda()
    model.eval()
    return model

# 2. 세그멘테이션 적용
def apply_segmentation(image, model):
    image_tensor = torch.tensor(image).permute(2, 0, 1).unsqueeze(0).float().cuda()
    height, width = image_tensor.shape[2], image_tensor.shape[3]
    padding = (0, (width + 31) // 32 * 32 - width, 0, (height + 31) // 32 * 32 - height)
    padded_image = F.pad(image_tensor, padding)

    with torch.no_grad():
        output = model(padded_image)
        mask = output.argmax(1).squeeze().cpu().numpy()

    return mask[:height, :width]  # 패딩 제거 후 원본 크기로 되돌림

# 3. 비 바다 영역의 RGB 값을 저장 (이미지 'C' 생성)
def store_non_ocean_rgb(original_image, segmented_mask):
    """
    세그멘테이션 결과로 비 바다 영역의 원본 RGB 값을 저장합니다.
    """
    non_ocean_mask = (segmented_mask == 0)  # 바다가 아닌 영역을 마스크로 생성
    non_ocean_rgb = np.zeros_like(original_image)  # 비 바다 영역을 저장할 배열 생성

    # 비 바다 영역에 해당하는 RGB 값을 저장
    non_ocean_rgb[non_ocean_mask] = original_image[non_ocean_mask]

    return non_ocean_rgb  # 이미지 'C' 반환

# 4. 최종 이미지 'D' 생성
def create_final_image(b_image_path, segmented_mask, non_ocean_rgb):
    """
    저장된 'B.png'에서 비 바다 영역을 제거한 후, 비 바다 영역의 원본 RGB 값을 덮어씌워 최종 이미지 'D'를 생성합니다.
    """
    # 'B.png' 이미지를 불러오고, RGBA 중 RGB만 사용
    b_image = Image.open(b_image_path).convert('RGB')
    final_image = np.array(b_image)  # alpha 채널 제거

    # 비 바다 영역의 마스크를 3차원으로 확장 (채널별로 적용)
    non_ocean_mask = (segmented_mask == 0)  # 바다가 아닌 영역의 마스크
    non_ocean_mask_3d = np.repeat(non_ocean_mask[:, :, np.newaxis], 3, axis=2)  # 2D 마스크를 3차원으로 확장

    # final_image와 non_ocean_rgb가 동일한 차원을 가지는지 확인
    if non_ocean_rgb.shape != final_image.shape:
        raise ValueError(f"non_ocean_rgb와 final_image의 차원이 일치하지 않습니다: {non_ocean_rgb.shape} vs {final_image.shape}")

    # 'B'에서 비 바다 영역(E)을 삭제한 후 'C'를 덮어씌움
    final_image[non_ocean_mask_3d] = non_ocean_rgb[non_ocean_mask_3d]

    return final_image  # 최종 이미지 'D'

# 5. 최종 시각화: 최종 이미지 'D'만 시각화
def visualize_results(final_image, save_path=None):
    """
    최종 결과 이미지 'D'만 시각화합니다.
    """
    plt.figure(figsize=(10, 10))
    plt.imshow(final_image)
    plt.title("Final Image (D)")
    plt.axis('off')  # 축 off

    plt.show()

    # 결과 저장 (저장 경로가 주어진 경우)
    if save_path:
        plt.savefig(os.path.join(save_path, 'final_image_for_web.png'), dpi=300, bbox_inches='tight')

# 6. main 함수 for web
def main(image_path, model_path, save_path):
    # 원본 이미지 'A' 로드
    original_image = np.array(Image.open(image_path).convert('RGB'))

    # 세그멘테이션 모델 로드 및 적용
    model = load_segmentation_model(model_path)
    segmented_mask = apply_segmentation(original_image, model)

    # 비 바다 영역의 RGB 값을 저장 (이미지 'C')
    non_ocean_rgb = store_non_ocean_rgb(original_image, segmented_mask)

    # K-means 클러스터링 및 히트맵 생성 후 저장 (이미지 'B.png' 생성)
    apply_kmeans_and_save_heatmap(original_image, save_path)  # Clustering.py에서 'B.png' 생성

    # 'B.png' 경로 설정
    b_image_path = os.path.join(save_path, 'B_web.png')

    # 최종 이미지 'D' 생성 (이미지 'B'에 비 바다 영역의 원본 RGB를 덮어씌움)
    final_image = create_final_image(b_image_path, segmented_mask, non_ocean_rgb)

    # 결과 시각화 및 저장
    visualize_results(final_image, save_path)

if __name__ == "__main__":
    image_path = '/home/work/yolotest/test.png'
    model_path = '/home/work/yolotest/unet_ocean_segmentation.pth'
    save_path = '/home/work/yolotest/results'
    main(image_path, model_path, save_path)
