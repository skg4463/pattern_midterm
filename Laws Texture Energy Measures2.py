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

def apply_laws_texture_energy(video_path, kernels):
    energy_maps = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
        energy_map = []
        for kernel in kernels:
            filtered = cv2.filter2D(gray, -1, kernel)
            energy_map.append(np.abs(filtered))
        energy_maps.append(energy_map)
    cap.release()
    return energy_maps

def visualize_energy_maps(energy_maps):
    fig, axes = plt.subplots(len(energy_maps), len(energy_maps[0]), figsize=(12, 12))
    for i in range(len(energy_maps)):
        for j in range(len(energy_maps[i])):
            axes[i, j].imshow(energy_maps[i][j], cmap='gray')
            axes[i, j].axis('off')
            axes[i, j].set_title(f'Energy Map ({i+1},{j+1})')
    plt.tight_layout()
    plt.show()

# 웹M 파일 경로 설정
video_path = '123.webm'

# 로스 텍스처 에너지 커널 생성
kernels = create_laws_kernels()

# 로스 텍스처 에너지 계산
energy_maps = apply_laws_texture_energy(video_path, kernels)

# 결과 시각화
visualize_energy_maps(energy_maps)
