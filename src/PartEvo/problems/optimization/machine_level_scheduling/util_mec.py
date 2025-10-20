import numpy as np
import matplotlib.pyplot as plt
import math


class PopulationInitializer:
    def __init__(self, best_solution, lower_bound, upper_bound, population_size):
        self.best_solution = best_solution
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.population_size = population_size
        self.dim = len(lower_bound)

    def initialize_population(self, methods):
        if self.best_solution is None:
            print('--->There is no initial best solution')
            methods = ['random']
            self.best_solution = (self.upper_bound + self.lower_bound) / 2


        num_methods = len(methods)
        method_weight = 1.0 / num_methods
        population_per_method = int(self.population_size * method_weight)

        population = []
        population.append(self.best_solution)
        for method in methods:
            if method == 'local_disturbance':
                population.extend(self.local_disturbance(population_per_method))
            elif method == 'random':
                population.extend(self.random_initialization(population_per_method))
            elif method == 'global_disturbance':
                population.extend(self.global_disturbance(population_per_method))
            elif method == 'boundary_initialization':
                population.extend(self.boundary_initialization(population_per_method))
            else:
                raise ValueError(f"Unknown initialization method: {method}")

        population = np.array(population)

        # Adjust population to match population_size
        if len(population) < self.population_size:
            # Fill with best_solution if population is insufficient
            additional_population = [self.best_solution] * (self.population_size - len(population))
            population = np.concatenate((population, additional_population), axis=0)
        elif len(population) > self.population_size:
            # Trim population if it exceeds the desired size
            population = population[:self.population_size]

        return population

    def local_disturbance(self, num_individuals):
        disturbance_population = []
        disturbance_range = 0.05
        for _ in range(num_individuals):
            disturbance = self.best_solution + np.random.uniform(-disturbance_range, disturbance_range,
                                                                 size=self.best_solution.shape)
            disturbance_population.append(self.clip_to_bounds(disturbance))
        return disturbance_population

    def random_initialization(self, num_individuals):
        random_population = []
        for _ in range(num_individuals):
            random_individual = np.random.uniform(self.lower_bound, self.upper_bound, size=self.best_solution.shape)
            random_population.append(random_individual)
        return random_population

    def global_disturbance(self, num_individuals):
        """
        Global disturbance initialization: generate individuals by applying a larger disturbance.
        """
        global_population = []
        disturbance_range = 0.3  # A larger range for global exploration
        for _ in range(num_individuals):
            disturbance = self.best_solution + np.random.uniform(-disturbance_range, disturbance_range,
                                                                 size=self.best_solution.shape)
            global_population.append(self.clip_to_bounds(disturbance))
        return global_population

    def boundary_initialization(self, num_individuals):
        """
        Boundary initialization: initialize individuals close to the lower and upper bounds.
        """
        boundary_population = []
        for i in range(num_individuals):
            if i % 2 == 0:
                boundary_individual = self.lower_bound + np.random.uniform(0, 0.2, size=self.best_solution.shape)
            else:
                boundary_individual = self.upper_bound - np.random.uniform(0, 0.2, size=self.best_solution.shape)
            boundary_population.append(self.clip_to_bounds(boundary_individual))
        return boundary_population

    def clip_to_bounds(self, solution):
        return np.clip(solution, self.lower_bound, self.upper_bound)

def select_by_ratio(lst, ratio):
    """
    按比例对应的索引，输出list的元素
    """
    # 确保索引不超过最后一个元素的索引
    index = min(math.floor(len(lst) * ratio), len(lst) - 1)
    return lst[index]


def initialize_mec_environment(n_devices, n_bases, width, height, radiu=300, plot_flag=True):
    """
    初始化地图，并返回移动设备、基站的坐标以及距离矩阵
    距离矩阵包括：
        移动设备 x 基站 的距离矩阵
        基站 x 基站 的距离矩阵
        移动设备 x 移动设备 的距离矩阵

    参数:
    - plot_flag: 控制是否绘制图形
    """
    np.random.seed(2024)
    # 动态计算行和列数，使得基站尽量均匀分布
    rows = int(np.ceil(np.sqrt(n_bases)))  # 计算行数
    cols = int(np.ceil(n_bases / rows))  # 计算列数，尽量让行列接近

    # 计算网格间距
    x_spacing = width / (cols + 1)  # 保证基站不在边缘
    y_spacing = height / (rows + 1)  # 保证基站不在边缘

    # 计算基站位置
    x_bases = np.linspace(x_spacing, width - x_spacing, cols)
    y_bases = np.linspace(y_spacing, height - y_spacing, rows)

    # 创建网格并获取基站的位置
    x_bases, y_bases = np.meshgrid(x_bases, y_bases)
    x_bases = x_bases.flatten()
    y_bases = y_bases.flatten()

    # 如果基站数量少于计算出来的点数，选择前n个基站
    x_bases = x_bases[:n_bases]
    y_bases = y_bases[:n_bases]

    # 将基站坐标转换为二维元组
    base_coords = [(x_bases[i], y_bases[i]) for i in range(n_bases)]

    # 计算基站的覆盖半径，取矩形的较小边长的50%
    coverage_radius = radiu

    # 初始化移动设备的位置，随机分布
    x_devices = np.random.uniform(0, width, n_devices)
    y_devices = np.random.uniform(0, height, n_devices)

    # 将移动设备坐标转换为二维元组
    device_coords = [(x_devices[i], y_devices[i]) for i in range(n_devices)]

    # 计算移动设备与基站的距离矩阵，行是移动设备，列是基站
    distances_devices = np.zeros((n_devices, n_bases))
    for i in range(n_devices):
        for j in range(n_bases):
            distances_devices[i, j] = np.sqrt((x_devices[i] - x_bases[j]) ** 2 + (y_devices[i] - y_bases[j]) ** 2)

    # 计算基站与基站之间的距离矩阵
    distances_bases = np.zeros((n_bases, n_bases))
    for i in range(n_bases):
        for j in range(i + 1, n_bases):
            distances_bases[i, j] = np.sqrt((x_bases[i] - x_bases[j]) ** 2 + (y_bases[i] - y_bases[j]) ** 2)
            distances_bases[j, i] = distances_bases[i, j]  # 对称矩阵

    # 计算设备与设备之间的距离矩阵
    distances_devices_devices = np.zeros((n_devices, n_devices))
    for i in range(n_devices):
        for j in range(i + 1, n_devices):
            distances_devices_devices[i, j] = np.sqrt(
                (x_devices[i] - x_devices[j]) ** 2 + (y_devices[i] - y_devices[j]) ** 2)
            distances_devices_devices[j, i] = distances_devices_devices[i, j]  # 对称矩阵

    # 如果绘制图形的标志为True，则进行绘图
    if plot_flag:
        plt.figure(figsize=(8, 8))
        plt.scatter(x_bases, y_bases, color='red', label='Base Stations')
        plt.scatter(x_devices, y_devices, color='blue', label='Mobile Devices')

        # 绘制覆盖半径 (假设从每个基站画一个圆)
        for i in range(n_bases):
            circle = plt.Circle((x_bases[i], y_bases[i]), coverage_radius, color='green', fill=False, linestyle='--')
            plt.gca().add_artist(circle)

        # 添加图例和标签
        plt.legend()
        plt.xlim(0, width)
        plt.ylim(0, height)
        plt.gca().set_aspect('equal', adjustable='box')
        plt.title('MEC Environment Initialization with Coverage Radius')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)

        plt.show()

    # 返回基站和移动设备的坐标，基站与移动设备的距离矩阵，基站与基站的距离矩阵，以及设备与设备的距离矩阵
    return base_coords, device_coords, distances_devices, distances_bases, distances_devices_devices

if __name__ == '__main__':
    # 用户输入矩形区域的宽度和高度
    width = 100  # 矩形区域的宽度
    height = 100  # 矩形区域的高度

    # 设置基站和移动设备的数量
    n_bases = 4  # 基站数量
    n_devices = 20  # 移动设备数量

    # 初始化MEC环境并返回坐标和距离矩阵
    base_coords, device_coords, distance_matrix_devices, distance_matrix_bases = initialize_mec_environment(n_bases,
                                                                                                            n_devices,
                                                                                                            width, height)

    # 打印基站和移动设备的坐标
    print("\nBase Stations Coordinates:", base_coords)
    print("\nMobile Devices Coordinates:", device_coords)

    # 打印基站与移动设备的距离矩阵
    print("\nDistance Matrix between Mobile Devices and Base Stations:")
    print(distance_matrix_devices)

    # 打印基站与基站的距离矩阵
    print("\nDistance Matrix between Base Stations:")
    print(distance_matrix_bases)
