import os
import re
import json
import matplotlib.pyplot as plt
import os

def read_tsptouropt(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith("COMMENT"):
                # 使用正则表达式匹配括号中的数字
                match = re.search(r'\((\d+)\)', line)
                if match:
                    return int(match.group(1))
    return None

def read_opttour(file_path):
    tour_list = []
    with open(file_path, 'r') as file:
        # 读取文件行
        lines = file.readlines()

        # 找到 TOUR_SECTION 行的位置
        tour_section_index = lines.index('TOUR_SECTION\n')

        # 遍历 TOUR_SECTION 之后的行，直到遇到 -1
        for line in lines[tour_section_index + 1:]:
            if '-1' in line:  # 遇到 -1 时停止读取
                break
            # 将每行的城市编号转换为整数并添加到列表中
            tour_list.extend(map(int, line.split()))

    return tour_list

def read_tsp2list(tspfilepath):
    nodes = []
    with open(tspfilepath, 'r') as file:
        lines = file.readlines()
        parsing_nodes = False
        for line in lines:
            line = line.strip()
            if line == 'EOF':
                break
            if line.startswith('NODE_COORD_SECTION'):
                parsing_nodes = True
                continue
            if parsing_nodes:
                parts = line.split()
                if len(parts) == 3:
                    node_id = int(parts[0])
                    x = float(parts[1])
                    y = float(parts[2])
                    nodes.append((node_id, x, y))
    return nodes

def save_optimal_tour_image(instance, opttour, save_directory='./instances_solution_png', **kwargs):
    """
    根据给定的 instance 名称和 opttour 路径，绘制最优路径并保存为 PNG 图像。

    参数:
    - instance: TSP instance 名称 (如 'ulysses16.tsp')
    - opttour: 最优路径的城市 ID 列表
    - save_directory: PNG 文件的保存目录，默认为 './instances_png'
    """
    # 生成文件保存路径
    tspinstance_name = instance.split('.tsp')[0]
    tsp_folder_path = kwargs.get('tsp_folder_path', './extracted_files/')
    tsp_totalpath = os.path.join(tsp_folder_path, instance)  # .tsp 文件的路径

    # 读取城市坐标
    tsp_city_coordinates = read_tsp2list(tsp_totalpath)  # 读取出的城市信息
    city_map = {city[0]: (city[1], city[2]) for city in tsp_city_coordinates}

    # 提取最优路径中的坐标
    opttour_coords = [city_map[city_id] for city_id in opttour]

    # 添加起始点作为路径的最后一个点形成闭环
    opttour_coords.append(opttour_coords[0])

    # 创建绘图
    plt.figure(figsize=(7, 7))

    # 绘制 opttour 路径
    x_coords, y_coords = zip(*opttour_coords)
    plt.plot(x_coords, y_coords, marker='o', linestyle='-', color='r')
    plt.title(f'Optimal Tour Path - {tspinstance_name}')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)

    # 标注城市ID
    for city_id, (x, y) in city_map.items():
        plt.text(x, y, str(city_id), fontsize=12, ha='center')

    # 确保保存目录存在
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # 保存图像为 PNG 文件
    save_path = os.path.join(save_directory, f'{tspinstance_name}_optimal_tour.png')
    plt.savefig(save_path)

    print(f'Optimal tour image saved as {save_path}')
    plt.close()  # 关闭图表，避免后续图形重叠

if __name__ == "__main__":
    test_instances = ['att48.tsp', 'berlin52.tsp', 'eil101.tsp', 'eil51.tsp', 'eil76.tsp',
                      'gr96.tsp', 'kroA100.tsp', 'kroC100.tsp', 'kroD100.tsp',
                      'pr76.tsp', 'ulysses16.tsp']  # 有tour

    tsp_folder_path = './extracted_files/'
    tsp_tour_path = './extracted_files/'
    tsp_png_save_catalog = './instances_png/'
    abspath = 'D:/00_Work/00_CityU/03_自动算法设计项目/MultiModal_exp'

    test_instances_tour = []
    test_instances_optcost = []
    extract_optcost = []
    llm_cost_save = []
    llm_tour = []
    llm_responds = []

    for instance in test_instances:
        """
        抽取tour中自带的最优cost
        """
        tspinstance_name = instance.split('.tsp')[0]
        tourpath = os.path.join(tsp_tour_path, tspinstance_name + ".opt.tour")
        opttour = read_opttour(tourpath)
        save_optimal_tour_image(instance, opttour,
                                save_directory='./instances_solution_png',
                                tsp_folder_path='./extracted_files/')

    print(extract_optcost)
