import numpy as np
from copy import deepcopy


class Environment:
    def __init__(self, I_set, P_set, np_set, T_set, data_init):
        self.I = I_set
        self.P = P_set
        self.n_p = np_set
        self.T = T_set
        self.hat_fsat = 1
        self.M_i = np.zeros((1, self.I))
        self.delta = 0.5
        self.U_i_p = deepcopy(data_init.U_i_p[:self.P, :self.I])
        self.CAP_p = deepcopy(data_init.CAP_p[:, :self.P])
        self.D_i_t = deepcopy(data_init.D_i_t[:self.T, :self.I])
        self.PAIR_i_i_p = deepcopy(data_init.PAIR_i_i_p[:self.I, :self.I])
        self.B_i_i_p_t = deepcopy(data_init.B_i_i_p_t[:self.I, :self.I])
        self.R_i_p_t = deepcopy(data_init.R_i_p_t[:self.P, :self.I])
        self.a_i_j_p = deepcopy(data_init.a_i_j_p[:self.P, :self.n_p, :self.I])
        self.b_i_j_p = deepcopy(data_init.b_i_j_p[:self.P, :self.n_p, :self.I])
        self.alpha_i_j_p = deepcopy(data_init.alpha_i_j_p[:self.P, :self.n_p, :self.I])
        self.TC_i_p_p = deepcopy(data_init.TC_i_p_p[:self.I, :self.P, :self.P])
        self.H2C_ip = deepcopy(data_init.H2C_ip[:self.P, :self.I])
        self.MLS_ipt = deepcopy(data_init.MLS_ipt[:self.P, :self.I])
        self.PMi = deepcopy(data_init.PMi[:, :self.I])
        self.Original_ip = deepcopy(data_init.Ori_ip[:self.P, :self.I])
        self.PCijpt = self.a_i_j_p + self.alpha_i_j_p * self.b_i_j_p
        self.boundary_1 = self.T * self.P * self.I
        self.boundary_2 = self.boundary_1 + self.T * self.P * (self.I - 8)
        self.boundary_3 = self.boundary_2 + self.I * self.P * self.T * (self.P - 1)
        self.boundary_4 = self.boundary_3 + self.I * self.P * self.T
        self.boundary_5 = self.boundary_4 + self.P * self.T * (self.I - 12) * (self.I - 13)
        print('粒子维度为：', self.boundary_5)
        # 上下限
        self.lower = np.zeros(self.boundary_5)
        self.upper = np.zeros(self.boundary_5)
        self.up_wijpt = self.n_p + 0.999  # 如果大于n_p，则说明不制造
        self.up_zipt = 60
        self.up_sippt = 30
        self.up_xipt = 30
        self.up_riipt = 1.1
        self.upper[:self.boundary_1] = self.up_wijpt
        self.upper[self.boundary_1: self.boundary_2] = self.up_zipt
        self.upper[self.boundary_2: self.boundary_3] = self.up_sippt
        self.upper[self.boundary_3: self.boundary_4] = self.up_xipt
        self.upper[self.boundary_4: self.boundary_5] = self.up_riipt

        #
        # self.upper[: self.boundary_1] = np.full((1, self.boundary_1), self.up_wijpt)[0]
        # # print('---', self.boundary_1, self.boundary_2, self.boundary_2 - self.boundary_1, self.up_zipt)
        # self.upper[self.boundary_1: self.boundary_2] = \
        #     np.full((1, self.boundary_2 - self.boundary_1), self.up_zipt)[0]
        # self.upper[self.boundary_2: self.boundary_3] = \
        #     np.full((1, self.boundary_3 - self.boundary_2), self.up_sippt)[0]
        # self.upper[self.boundary_3: self.boundary_4] = \
        #     np.full((1, self.boundary_4 - self.boundary_3), self.up_xipt)[0]
        # self.upper[self.boundary_4: self.boundary_5] = \
        #     np.full((1, self.boundary_5 - self.boundary_4), self.up_riipt)[0]
        self.penalty_param = 100000

        self.inited_positions = self.position_init(30)


    def position_to_matrix(self, input_position):
        position_to_matrix = np.trunc(input_position).copy()
        wijpt = np.zeros((self.T, self.P, self.n_p, self.I))
        zipt = np.zeros((self.T, self.P, self.I))
        sippt = np.zeros((self.T, self.I, self.P, self.P))
        xipt = np.zeros((self.T, self.P, self.I))
        riipt = np.zeros((self.T, self.P, self.I, self.I))
        # load wijpt
        position_con = 0
        for wijpt_t in range(self.T):
            for wijpt_p in range(self.P):
                for wijpt_i in range(self.I):
                    if position_to_matrix[position_con] < self.n_p:
                        wijpt[wijpt_t, wijpt_p, int(position_to_matrix[position_con]), wijpt_i] = 1
                    position_con += 1
        # load zipt
        for zipt_t in range(self.T):
            for zipt_p in range(self.P):
                for zipt_i in range(8, self.I):
                    zipt[zipt_t, zipt_p, zipt_i] = position_to_matrix[position_con]
                    position_con += 1
        # load sippt
        for sippt_t in range(self.T):
            for sippt_i in range(self.I):
                for sippt_p1 in range(self.P):
                    for sippt_p2 in range(self.P):
                        if sippt_p1 == sippt_p2:
                            continue
                        sippt[sippt_t, sippt_i, sippt_p2, sippt_p1] = position_to_matrix[position_con]
                        position_con += 1
        # load xipt
        for xipt_t in range(self.T):
            for xipt_p in range(self.P):
                for xipt_i in range(self.I):
                    xipt[xipt_t, xipt_p, xipt_i] = position_to_matrix[position_con]
                    position_con += 1
        # load riipt
        for riipt_t in range(self.T):
            for riipt_p in range(self.P):
                for riipt_i1 in range(12, self.I):
                    for riipt_i2 in range(12, self.I):
                        if riipt_i1 == riipt_i2:
                            continue
                        riipt[riipt_t, riipt_p, riipt_i2, riipt_i1] = position_to_matrix[position_con]
                        position_con += 1
        return wijpt, zipt, sippt, xipt, riipt

    def objfunction(self, position_cal):
        # print('shape_position', np.shape(position_cal))
        inequality = 0
        cost_write = np.zeros((3, 1))
        wijpt_obj, zipt_obj, sippt_obj, xipt_obj, riipt_obj = self.position_to_matrix(position_cal)
        Pcost, hxipt_paired = self.cal_Pcost(wijpt_obj, xipt_obj)
        Tcost = self.cal_Tcost(sippt_obj)
        Hcost = self.cal_Hcost(zipt_obj)
        cost_write[0, 0] = Pcost
        cost_write[1, 0] = Tcost
        cost_write[2, 0] = Hcost
        Totalcost = Pcost + Tcost + Hcost
        Store_ipt, wipt_obj = self.cal_storeipt(wijpt_obj, hxipt_paired, sippt_obj, riipt_obj, zipt_obj)
        # con 1
        fsati_obj, mit_obj, uit_obj = self.constraint_1(zipt_obj)
        # inequality += np.sum(np.maximum(self.hat_fsat - fsati_obj, 0))
        inequality += np.sum(np.maximum(self.hat_fsat - uit_obj, 0))
        # con 2
        inequality += np.sum(np.maximum(np.sum(Store_ipt * self.U_i_p, axis=2) - self.CAP_p[0], 0))
        # con 3
        # con 4
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - (np.dot(hxipt_paired * wipt_obj, self.B_i_i_p_t.T) + zipt_obj), 0))
        # con 5
        inequality += np.sum(np.maximum(np.sum(riipt_obj, axis=2) - self.R_i_p_t, 0))
        # con 6
        # con 7
        inequality += np.sum(np.maximum(-Store_ipt, 0))
        penalty = inequality * self.penalty_param
        obj = penalty + Totalcost
        return obj

    def objfunction_final(self, position_cal):
        inequality = 0
        cost_write = np.zeros((3, 1))
        wijpt_obj, zipt_obj, sippt_obj, xipt_obj, riipt_obj = self.position_to_matrix(position_cal)
        Pcost, hxipt_paired = self.cal_Pcost(wijpt_obj, xipt_obj)
        Tcost = self.cal_Tcost(sippt_obj)
        Hcost = self.cal_Hcost(zipt_obj)
        cost_write[0, 0] = Pcost
        cost_write[1, 0] = Tcost
        cost_write[2, 0] = Hcost
        Totalcost = Pcost + Tcost + Hcost
        Store_ipt, wipt_obj = self.cal_storeipt(wijpt_obj, hxipt_paired, sippt_obj, riipt_obj, zipt_obj)
        # con 1
        fsati_obj, mit_obj, uit_obj = self.constraint_1(zipt_obj)
        # con_1 = np.sum(np.maximum(self.hat_fsat - fsati_obj, 0))
        con_1 = np.sum(np.maximum(self.hat_fsat - uit_obj, 0))
        inequality += con_1
        # con 2
        con_2 = np.sum(np.maximum(np.sum(Store_ipt * self.U_i_p, axis=2) - self.CAP_p[0], 0))
        inequality += con_2
        # con 3
        # con 4
        con_4 = np.sum(np.maximum(np.sum(riipt_obj, axis=2) - (np.dot(hxipt_paired * wipt_obj, self.B_i_i_p_t.T) + zipt_obj), 0))
        inequality += con_4
        # con 5
        con_5 = np.sum(np.maximum(np.sum(riipt_obj, axis=2) - self.R_i_p_t, 0))
        inequality += con_5
        # con 6
        # con 7
        con_7 = np.sum(np.maximum(-Store_ipt, 0))
        inequality += con_7
        penalty = inequality * self.penalty_param
        obj = penalty + Totalcost
        con_sum = np.array([[con_1, con_2, con_4, con_5, con_7]])
        return obj, Totalcost, penalty, fsati_obj, con_sum, cost_write, mit_obj, uit_obj

    def cal_Pcost(self, wijpt_pcost, xipt_pcost):
        PCipt = np.sum(self.PCijpt * wijpt_pcost, axis=2)
        hxipt = xipt_pcost * self.PMi
        hxipt_pair = self.constrain_3(hxipt)
        # print('test_1', hxipt_pair[0])
        main_hxipt = hxipt_pair > self.MLS_ipt
        hxipt_pair = hxipt_pair * main_hxipt
        pcost = np.sum(PCipt * hxipt_pair)
        return pcost, hxipt_pair

    def cal_Tcost(self, sippt_tcost):
        sipp = np.sum(sippt_tcost, axis=0)
        tcost = np.sum(self.TC_i_p_p * sipp)
        return tcost

    def cal_Hcost(self, zipt_hcost):
        zip_hcost = np.sum(zipt_hcost, axis=0)
        hcost = np.sum(zip_hcost * self.H2C_ip)
        return hcost

    def constrain_3(self, hxipt_input):
        hxipt_pair = deepcopy(hxipt_input)
        if np.shape(hxipt_pair)[2] >= 13:
            hxipt_pair[:, :, 12] = hxipt_pair[:, :, 10] * 4
        if np.shape(hxipt_pair)[2] >= 19:
            hxipt_pair[:, :, 18] = hxipt_pair[:, :, 19]
        return hxipt_pair

    def cal_storeipt(self, wijpt_store, hxipt_paired, sippt_store, riipt_store, zipt_store):
        wipt_store = np.sum(wijpt_store, axis=2)
        increment_ipt = hxipt_paired * wipt_store + np.swapaxes(np.sum(sippt_store, axis=2), 1, 2) + np.sum(riipt_store, axis=2)
        decrement_ipt = zipt_store + np.swapaxes(np.sum(sippt_store, axis=3), 1, 2) + np.sum(riipt_store,
                                                                                             axis=3) + np.dot(
            hxipt_paired * wipt_store, self.B_i_i_p_t.T)
        store_ipt = np.zeros((self.T, self.P, self.I))
        store_ipt[0, :, :] = self.Original_ip[:, :] + increment_ipt[0, :, :] - decrement_ipt[0, :, :]
        for t_store in range(1, self.T):
            store_ipt[t_store, :, :] = store_ipt[t_store - 1, :, :] + increment_ipt[t_store, :, :] - decrement_ipt[
                                                                                                     t_store, :, :]
        return store_ipt, wipt_store

    def constraint_1(self, zipt_c1):
        mit_before = np.zeros((self.T, self.I))
        zit = np.sum(zipt_c1, axis=1)
        mit_before[0, :] = self.M_i[0, :] + self.D_i_t[0, :] - zit[0, :]
        for t_c1 in range(1, self.T):
            mit_before[t_c1, :] = mit_before[t_c1 - 1, :] + self.D_i_t[t_c1, :] - zit[t_c1, :]
        mit = np.maximum(mit_before, 0)
        uit_before = np.zeros((self.T, self.I))
        uit_before[0, :] = (zit[0, :] - self.M_i[0] + self.delta) / (self.D_i_t[0] + self.delta)
        for t_c1 in range(1, self.T):
            uit_before[t_c1, :] = (zit[t_c1, :] - mit[t_c1 - 1, :] + self.delta) / (self.D_i_t[t_c1, :] + self.delta)
        uit = np.maximum(uit_before, 0)
        fsati = np.sum(uit, axis=0) / self.T
        # print(fsati)
        # print(np.shape(fsati))
        return fsati, mit, uit

    def position_init(self, numofparticle):
        position_init = np.zeros((numofparticle, self.boundary_5))
        for num_init in range(numofparticle):
            position_init[num_init, 0:self.boundary_1] = np.random.randint(0, self.up_wijpt, (1, self.boundary_1))[0, :]
            position_init[num_init, self.boundary_1: self.boundary_2] = np.random.randint(0, self.up_zipt, (
                1, self.boundary_2 - self.boundary_1))[0, :]
            position_init[num_init, self.boundary_2: self.boundary_3] = np.random.randint(0, self.up_sippt, (
                1, self.boundary_3 - self.boundary_2))[0, :]
            position_init[num_init, self.boundary_3: self.boundary_4] = np.random.randint(0, self.up_xipt, (
                1, self.boundary_4 - self.boundary_3))[0, :]
            position_init[num_init, self.boundary_4: self.boundary_5] = np.random.randint(0, self.up_riipt, (
                1, self.boundary_5 - self.boundary_4))[0, :]
        return position_init


if __name__ == '__main__':
    from datainit_i30_p10_t30_np10 import Dataenv
    data_I = Dataenv()
    a = Environment(21, 6, 6, 20, data_I)
    # print(a.U_i_p)
    # print(np.shape(a.U_i_p))
    # print(a.CAP_p)
    # print(a.D_i_t)
    # print(np.shape(a.D_i_t))
    # print(a.PAIR_i_i_p)
    # print(a.B_i_i_p_t)
    # print(a.R_i_p_t)
    # print(np.shape(a.a_i_j_p))
    # print(np.shape(a.b_i_j_p))
    # print(np.shape(a.alpha_i_j_p))
    # print(a.TC_i_p_p)
    # print(a.PMi)
    # print(np.shape(a.D_i_t))
    # print(type(a.D_i_t))
    # print(a.up_of_position)
    # print(a.Original_ip)
    # test_position = np.arange(a.boundary_5)
    # test_position_2 = np.ones(a.boundary_5) * 12
    # test_position_2[:a.boundary_1] = 1
    # # print(np.shape(test_position))
    # # print(test_position)
    # # print(a.position_to_matrix(test_position)[-1, -1])
    # # print(np.shape(a.position_to_matrix(test_position)))
    # a.objfunction(test_position_2)
    P_inti = a.position_init(1)
    print(np.shape(P_inti))
    print(P_inti)
