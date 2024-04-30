import matplotlib.pyplot as plt
from skimage import io
from sklearn.cluster import MeanShift, estimate_bandwidth

def load_and_normalize_image(image_path):
    # 이미지를 로드하고 0-1 범위로 정규화
    image = io.imread(image_path)
    return image / 255.0

def apply_meanshift(pixels, quantile=0.1):
    # 대역폭 추정
    bandwidth = estimate_bandwidth(pixels, quantile=quantile, n_samples=500)
    # 평균 이동 클러스터링 수행
    meanshift = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(pixels)
    # 각 픽셀의 클러스터 레이블을 반환
    return meanshift.labels_

def plot_meanshift_results(image_shape, meanshift_labels):
    # 클러스터링 결과를 시각화
    plt.figure(figsize=(8, 8))
    plt.imshow(meanshift_labels.reshape(image_shape[0], image_shape[1]), cmap='viridis')
    plt.title('Mean Shift Image Segmentation')
    plt.axis('off')  # 축 라벨 제거
    plt.show()

def main():
    image_path = 'suwon.jpg'  # 이미지 경로 설정
    image = load_and_normalize_image(image_path)
    pixels = image.reshape(-1, 3)  # 이미지를 평탄화

    meanshift_labels = apply_meanshift(pixels)
    plot_meanshift_results(image.shape, meanshift_labels)

if __name__ == "__main__":
    main()
