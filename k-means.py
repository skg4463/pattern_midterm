import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import KMeans

def load_and_normalize_image(image_path):
    # 이미지를 로드하고 0-1 범위로 정규화
    image = io.imread(image_path)
    return image / 255.0

def apply_kmeans(pixels, n_clusters=4):
    # K-평균 클러스터링을 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    # 각 픽셀의 클러스터 레이블을 반환
    return kmeans.labels_

def plot_kmeans_results(image_shape, kmeans_labels):
    # 클러스터링 결과를 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(kmeans_labels.reshape(image_shape[0], image_shape[1]), cmap='viridis')
    plt.title('K-Means Image Segmentation')
    plt.axis('off')  # 축 라벨 제거
    plt.show()

def main():
    image_path = 'suwon.jpg'  # 이미지 경로 설정
    image = load_and_normalize_image(image_path)
    pixels = image.reshape(-1, 3)  # 이미지를 평탄화

    kmeans_labels = apply_kmeans(pixels)
    plot_kmeans_results(image.shape, kmeans_labels)

if __name__ == "__main__":
    main()
