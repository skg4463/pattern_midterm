import matplotlib.pyplot as plt
import numpy as np
from skimage import io
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def load_and_normalize_image(image_path):
    # 이미지를 로드하고 0-1 범위로 정규화
    image = io.imread(image_path)
    return image / 255.0


def apply_kmeans(pixels, n_clusters=4):
    # K-평균 클러스터링을 수행
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(pixels)
    return kmeans.labels_


def estimate_prior_probabilities(labels, n_clusters):
    # 각 클러스터에 대한 사전 확률을 계산
    total_count = len(labels)
    return [np.sum(labels == i) / total_count for i in range(n_clusters)]


def apply_gaussian_mixture(pixels, kmeans_labels, n_clusters):
    # 클러스터 레이블을 사용하여 가우시안 혼합 모델을 초기화하고 적합
    gmm = GaussianMixture(n_components=n_clusters, covariance_type='full', random_state=0)
    gmm.fit(pixels, kmeans_labels)
    return gmm.predict(pixels)


def plot_classification_results(image_shape, gmm_labels):
    # 분류 결과를 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(gmm_labels.reshape(image_shape[0], image_shape[1]), cmap='viridis')
    plt.title('Classification with MLE')
    plt.axis('off')  # 축 라벨 제거
    plt.show()


def main():
    image_path = 'suwon.jpg'  # 이미지 경로 설정
    image = load_and_normalize_image(image_path)
    pixels = image.reshape(-1, 3)  # 이미지를 평탄화

    n_clusters = 4  # 클러스터 수 설정
    kmeans_labels = apply_kmeans(pixels, n_clusters)
    prior_probabilities = estimate_prior_probabilities(kmeans_labels, n_clusters)
    print("Prior probabilities of each cluster:", prior_probabilities)

    gmm_labels = apply_gaussian_mixture(pixels, kmeans_labels, n_clusters)
    plot_classification_results(image.shape, gmm_labels)


if __name__ == "__main__":
    main()
