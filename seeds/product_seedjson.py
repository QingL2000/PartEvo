"""
该脚本用于将改脚本所在目录下human_algorithms的各算法py文件中的算法存成json
"""

import os
import importlib.util
import inspect
import json
import re


def load_algos_from_directory(directory):
    algo_list = []
    print(os.listdir(directory))
    for filename in os.listdir(directory):
        if filename.endswith('.py'):
            file_path = os.path.join(directory, filename)
            module_name = os.path.splitext(filename)[0]

            # 读取整个文件内容
            with open(file_path, 'r', encoding='utf-8') as file:
                file_content = file.read()

            # 提取 import 语句
            import_statements = []
            for line in file_content.splitlines():
                if line.startswith('import ') or line.startswith('from '):
                    import_statements.append(line)

            # 动态加载模块
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            if hasattr(module, 'algo'):
                algo_func = module.algo
                algo_code = inspect.getsource(algo_func)

                # 移除描述部分
                algo_code = re.sub(r'""".*?"""', '', algo_code, flags=re.DOTALL).strip()

                docstring = algo_func.__doc__ if algo_func.__doc__ else "No description"

                algo_list.append({
                    'algorithm': docstring.strip(),
                    'code': '\n'.join(import_statements) + '\n' + algo_code,
                    "objective": None,
                    "other_inf": None
                })

    return algo_list


def dict2json(json_file_path):
    with open(json_file_path, 'w', encoding='utf-8') as json_file:
        json.dump(algorithms, json_file, ensure_ascii=False, indent=4)

    print(f"Algorithms have been saved to {json_file_path}")


if __name__ == '__main__':

    current_file_path = os.path.abspath(__file__)
    # 指定目录路径
    # directory_path = 'D:\\00_Work\\00_CityU\\04_AEL_MEC\\test_code\\test_function'
    directory_path = os.path.join(os.path.dirname(current_file_path), 'human_algorithms')
    print(directory_path)

    # 加载算法
    algorithms = load_algos_from_directory(directory_path)
    # print(algorithms)

    # 打印结果
    for algo in algorithms:
        print(algo)
        # print(f"Algorithm: {algo_name}")
        # print(f"Description: {algo_info['algorithm']}")
        # print(f"Code:\n{algo_info['code']}\n")
        print()

    json_file_path = 'seed0.json'
    dict2json(json_file_path)
