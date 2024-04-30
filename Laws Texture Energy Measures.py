import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

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

def save_energy_maps(energy_maps, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for i, frame_energy_maps in enumerate(energy_maps):
        for j, energy_map in enumerate(frame_energy_maps):
            plt.imsave(os.path.join(output_folder, f'frame_{i}_energy_map_{j}.png'), energy_map, cmap='gray')

# 텍스처 웹M 파일 경로 설정
texture_video_path = '123.webm'

# 로스 텍스처 에너지 커널 생성
kernels = create_laws_kernels()

# 로스 텍스처 에너지 계산
energy_maps = apply_laws_texture_energy(texture_video_path, kernels)

# 계산된 에너지 맵 저장
output_folder = 'energy_maps'
save_energy_maps(energy_maps, output_folder)
