import random


def greedy(current_objective, best_objective):
    return current_objective < best_objective


# 示例选择策略：模拟退火（简单版本）
def simulated_annealing_strategy(current_objective, best_objective, temperature):
    if current_objective < best_objective:
        return True
    else:
        probability = random.random()
        return probability < temperature  # 依赖温度来决定是否接受较差的解


class Branch:
    def __init__(self, branchno):
        self.branchno = branchno
        self.algorithm = ""
        self.code = ""
        self.objective = None
        self.local_algorithm = ""
        self.local_code = ""
        self.local_objective = None
        self.branch = None
        self.thought = ""
        self.history_option = ""
        self.history_thought = ""
        self.selection_strategy = greedy
        self.reflection = ""

        options = ['Init', 'ie', 'ce', 'ge', 'pe']
        contents = ['algorithm', 'code', 'objective']
        self.opresult_recorder = {option: {content: None for content in contents} for option in options}

    def update_opresult_recoder(self, op, offspring_dict):
        self.opresult_recorder[op]['algorithm'] = offspring_dict['algorithm']
        self.opresult_recorder[op]['code'] = offspring_dict['code']
        self.opresult_recorder[op]['objective'] = offspring_dict['objective']
        if self.opresult_recorder[op]['objective']:
            if self.opresult_recorder[op]['objective'] < self.local_objective:
                self.local_algorithm = self.opresult_recorder[op]['algorithm']
                self.local_code = self.opresult_recorder[op]['code']
                self.local_objective = self.opresult_recorder[op]['objective']

    def selection_in_branch(self):
        best_option = None
        best_objective = self.objective  # 初始化为正无穷，便于寻找最小值

        for option in ['ie', 'ce', 'ge', 'pe']:
            objective = self.opresult_recorder[option]['objective']
            if objective is not None:
                if self.selection_strategy(objective, best_objective):
                    best_objective = objective
                    best_option = option

        if best_option:
            self.algorithm = self.opresult_recorder[best_option]['algorithm']
            self.code = self.opresult_recorder[best_option]['code']
            self.objective = best_objective
            print(f'Branch {self.branchno} update by option {best_option}')

    def init_branch(self, **kwargs):
        if kwargs.get('ind_in_dict'):
            ind_in_dict = kwargs.get('ind_in_dict')
            self.algorithm = ind_in_dict['algorithm']
            self.code = ind_in_dict['code']
            self.objective = ind_in_dict['objective']
        elif kwargs.get('algorithm') and kwargs.get('code') and kwargs.get('objective'):
            self.algorithm = kwargs.get('algorithm')
            self.code = kwargs.get('code')
            self.objective = kwargs.get('objective')
        else:
            print('Here is no suitable input to build a branch')
            exit()
        self.opresult_recorder['Init']['algorithm'] = self.algorithm
        self.opresult_recorder['Init']['code'] = self.code
        self.opresult_recorder['Init']['objective'] = self.objective
        self.local_algorithm = self.algorithm
        self.local_code = self.code
        self.local_objective = self.objective

    def branch_to_dict(self):
        return {'algorithm': self.algorithm,
                'code': self.code,
                'objective': self.objective,
                'other_inf': None}

    def set_algorithm(self, algorithm):
        self.algorithm = algorithm

    def set_reflection(self, reflection):
        self.reflection = reflection

    def set_code(self, code):
        self.code = code

    def set_objective(self, objective):
        self.objective = objective

    def set_local_algorithm(self, local_algorithm):
        self.local_algorithm = local_algorithm

    def set_local_code(self, local_code):
        self.local_code = local_code

    def set_branch(self, branch):
        self.branch = branch

    def set_thought(self, thought):
        self.thought = thought

    def add_history_option(self, history_option):
        self.history_option += history_option
        self.history_option += '_'

    def add_history_thought(self, history_thought):
        self.history_thought += history_thought
        self.history_thought += '\n'


def get_random_cooperator_branches(input_list, index, m):
    if m > len(input_list):
        raise ValueError("m must be less than or equal to the total number of elements.")

    # 获取所有元素的索引
    choices = list(range(len(input_list)))

    # 从choices中排除指定的索引
    choices.remove(index)

    # 随机选择m-1个与index不同的索引
    selected_indices = random.sample(choices, m)

    # 将输入索引添加到返回的索引列表的最前面
    selected_indices.insert(0, index)

    # 根据索引返回对应的元素
    # print("The indexs seleted are", selected_indices)
    return [input_list[i] for i in selected_indices]
