import matplotlib
matplotlib.use('Agg')  # GUI 없는 백엔드 설정
import numpy as np
import matplotlib.pyplot as plt
import cv2
from matplotlib.animation import FuncAnimation

class Visualizer:
    def __init__(self, image_path, depth_map_path):
        self.image_path = image_path
        self.depth_map_path = depth_map_path
        self.image = None
        self.depth_map = None
        self.x = None
        self.y = None
        self.z = None
        self.colors = None

    def load_data(self):
        self.image = cv2.imread(self.image_path)
        self.depth_map = cv2.imread(self.depth_map_path, cv2.IMREAD_GRAYSCALE)

        if self.image is None or self.depth_map is None:
            raise ValueError("이미지 또는 깊이 맵을 로드할 수 없습니다.")

        height, width, _ = self.image.shape
        depth_map_resized = cv2.resize(self.depth_map, (width, height), interpolation=cv2.INTER_LINEAR)

        self.x = np.arange(0, width)
        self.z = np.arange(0, height)
        self.x, self.z = np.meshgrid(self.x, self.z)

        # 깊이 값을 max - 깊이값으로 변환
        self.y = depth_map_resized.astype(float)
        self.y = np.max(self.y) - self.y  # max - 깊이값

        image_rgb = cv2.cvtColor(self.image, cv2.COLOR_BGR2RGB)
        self.colors = image_rgb / 255.0

    def visualize(self, save_animation=False, output_path="rotating_xy_plane.mp4"):
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        scatter = ax.scatter(self.x, self.y, self.z, c=self.colors.reshape(-1, 3), s=0.5)
        ax.grid(False)
        ax.axis('off')
        ax.dist = 1

        def update(frame):
            ax.view_init(elev=190, azim=frame)
            return scatter,

        ani = FuncAnimation(fig, update, frames=np.arange(0, 360, 15), interval=100, blit=False)

        if save_animation:
            ani.save(output_path, writer='ffmpeg', fps=5)
        else:
            plt.show()


# 사용 예시
if __name__ == "__main__":
    image_path = "static/uploads/2.jpg"  # 원본 이미지 경로
    depth_map_path = "static/generated/predicted_2.jpg"  # 깊이 맵 경로

    # Visualizer 객체 생성 및 사용
    visualizer = Visualizer(image_path, depth_map_path)
    visualizer.load_data()
    visualizer.visualize(save_animation=False)
