# from machinelearning import *
# from mathematics import *
# from optimization import *
# from physics import *
class Probs():
    def __init__(self, paras):

        if not isinstance(paras.problem, str):
            self.prob = paras.problem
            print("- Prob local loaded ")
        elif paras.problem == "tsp_construct":
            from .optimization.tsp_greedy import run
            self.prob = run.TSPCONST()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == 'tsp_gls':
            from .optimization.tsp_gls import run
            self.prob = run.TSPGLS()
            print("- Prob " + paras.problem + " loaded ")

        elif paras.problem == "mec_task_offloading":
            from .optimization.mec_task_offloading import run
            self.prob = run.MECENV()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "mec_task_offloading_new":
            from .optimization.mec_task_offloading_new import run
            self.prob = run.MECENV()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "mec_task_offloading_blackbox":
            from .optimization.mec_task_offloading_blackbox import run
            self.prob = run.MECENV()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "single_mode":
            from .optimization.single_mode_blackbox import run
            self.prob = run.Baseline()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "multi_mode":
            from .optimization.multi_mode_blackbox import run
            self.prob = run.Baseline_multi()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "machine_level_scheduling":
            from .optimization.machine_level_scheduling import run
            self.prob = run.MLSENV()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "bp_online":
            from .optimization.bp_online import run
            self.prob = run.BPONLINE()
            print("- Prob " + paras.problem + " loaded ")
        elif paras.problem == "bp_online_llm4ad":
            from .optimization.bp_online_llm4ad import run
            self.prob = run.BPONLINE()
            print("- Prob " + paras.problem + " loaded ")
        else:
            print("problem " + paras.problem + " not found!")

    def get_problem(self):

        return self.prob
