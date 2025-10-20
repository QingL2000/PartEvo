from .individual_cluster import Individual
import random



class ExternalSet:
    """
    从小到大
    """

    def __init__(self, limit):
        self.solutions = []
        self.limit = limit

    def add_solution(self, solution):
        # 如果 solution 是 Branch 对象，则转换为字典
        if isinstance(solution, Individual):
            solution_dict = {'algorithm': solution.algorithm,
                             'code': solution.code,
                             'objective': solution.objective,
                             'branch': solution.thought}
        else:
            solution_dict = solution

        if len(self.solutions) < self.limit:
            # 如果当前解集未满，直接添加
            self.solutions.append(solution_dict)
            self.solutions.sort(key=lambda x: x['objective'])
        else:
            # 如果当前解集已满，检查新解的 objective
            max_solution = max(self.solutions, key=lambda x: x['objective'])
            if solution_dict['objective'] < max_solution['objective']:
                self.solutions.remove(max_solution)
                self.solutions.append(solution_dict)
                self.solutions.sort(key=lambda x: x['objective'])

    def remove_solution(self, solution):
        if solution in self.solutions:
            self.solutions.remove(solution)
            self.solutions.sort(key=lambda x: x['objective'])

    def get_solutions(self):
        return self.solutions

    def get_best_solution(self):
        return self.solutions[0] if self.solutions else None

    def __len__(self):
        return len(self.solutions)

    def __iter__(self):
        self._iter_index = 0
        return self

    def __next__(self):
        if self._iter_index < len(self.solutions):
            solution = self.solutions[self._iter_index]
            self._iter_index += 1
            return solution
        else:
            raise StopIteration

    def get_solutions_for_summary(self, threshold_size=10):
        # 如果解的个数小于 threshold_size，则直接返回所有解
        if len(self.solutions) < threshold_size:
            return self.solutions
        else:
            # 从大到小均匀采样 threshold_size 个解
            step = len(self.solutions) / threshold_size  # 计算浮动步长
            sampled_solutions = []

            for i in range(threshold_size):
                # 计算每个解的索引，并取整
                index = int(i * step)
                sampled_solutions.append(self.solutions[index])

            return sampled_solutions

if __name__ == "__main__":
    a = ExternalSet(10)
    print(len(a))
