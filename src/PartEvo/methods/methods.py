from .selection import prob_rank, equal, roulette_wheel, tournament
from .management import pop_greedy, ls_greedy, ls_sa


class Methods():
    def __init__(self, paras, problem) -> None:
        """
        selection 和 manage相当于遗传中的 父辈选择 和 种群维护

        selection(pop, m)
        负责从当前的种群pop中选择进行进化的m个父辈个体
        输入的pop(population)是一个列表，列表中每个元素是一个个体
        每个个体是一个dict

        manage(pop,size)
        从当前种群pop中筛选size个个体作为下一代
        """
        self.paras = paras
        self.problem = problem
        if paras.selection == "prob_rank":
            self.select = prob_rank
        elif paras.selection == "equal":
            self.select = equal
        elif paras.selection == 'roulette_wheel':
            self.select = roulette_wheel
        elif paras.selection == 'tournament':
            self.select = tournament
        else:
            print("selection method " + paras.selection + " has not been implemented !")
            exit()

        if paras.management == "pop_greedy":
            self.manage = pop_greedy
        elif paras.management == 'ls_greedy':
            self.manage = ls_greedy
        elif paras.management == 'ls_sa':
            self.manage = ls_sa
        else:
            print("management method " + paras.management + " has not been implemented !")
            exit()

    def get_method(self):

        if self.paras.method == "ael":
            from .ael.ael import AEL
            return AEL(self.paras, self.problem, self.select, self.manage, psoparams=self.paras.psoparams)
        elif self.paras.method == 'eoh':
            from .eoh.eoh import EOH
            return EOH(self.paras, self.problem, self.select, self.manage, psoparams=self.paras.psoparams)
        elif self.paras.method == "sie":
            from .sie.sie import SIE
            return SIE(self.paras, self.problem, self.select, self.manage)
        elif self.paras.method == 'partevo':
            from .partevo.partevo import PartEvo
            return PartEvo(self.paras, self.problem, self.select, self.manage)
        else:
            print("method " + self.method + " has not been implemented!")
            exit()
