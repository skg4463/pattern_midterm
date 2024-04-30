import numpy as np
import cv2
import matplotlib.pyplot as plt

def create_laws_kernels():
    L5 = np.array([1, 4, 6, 4, 1])  # Level
    E5 = np.array([-1, -2, 0, 2, 1])  # Edge
    S5 = np.array([-1, 0, 2, 0, -1])  # Spot
    W5 = np.array([-1, 2, 0, -2, 1])  # Wave
    R5 = np.array([1, -4, 6, -4, 1])  # Ripple

    kernels = []
    vectors = [L5, E5, S5, W5, R5]
    for i in vectors:
        for j in vectors:
            kernels.append(np.outer(i, j))
    return kernels

def apply_laws_texture_energy(image, kernels):
    energy_maps = []
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    for kernel in kernels:
        filtered = cv2.filter2D(gray, -1, kernel)
        energy_maps.append(np.abs(filtered))
    return energy_maps

def visualize_energy_maps(energy_maps):
    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    for i, ax in enumerate(axes.flatten()):
        ax.imshow(energy_maps[i], cmap='gray')
        ax.axis('off')
        ax.set_title(f'Energy Map {i+1}')
    plt.tight_layout()
    plt.show()

# 이미지 파일 경로 설정
image_path = '104591554.4.jpg'

# 로스 텍스처 에너지 커널 생성
kernels = create_laws_kernels()

# 이미지 읽기
image = cv2.imread(image_path)

# 로스 텍스처 에너지 계산
energy_maps = apply_laws_texture_energy(image, kernels)

# 결과 시각화
visualize_energy_maps(energy_maps)
