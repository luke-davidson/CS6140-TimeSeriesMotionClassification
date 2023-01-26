import pandas as pd
import numpy as np
import random
import time
import matplotlib.pyplot as plt
import statistics

class Analyzer():
    def __init__(self, all_data_norm_df):
        self.all_data_norm = all_data_norm_df
        self.upper_limbs = ["l_arm_m2_", "l_arm_m3_", "r_arm_m4_", "r_arm_m5_"]
        self.lower_limbs = ["l_leg_m6_", "l_leg_m7_", "r_leg_m8_", "r_leg_m9_"]
        self.head = ["head_"]
        self.all_limbs = self.upper_limbs + self.lower_limbs + self.head
        self.best_limbs = {
            "Bowing": self.upper_limbs + self.head, 
            "Clapping": self.upper_limbs, 
            "Handshaking": self.upper_limbs, 
            "Hugging": self.upper_limbs, 
            "Jumping": self.lower_limbs, 
            "Running": self.all_limbs, 
            "Seating": self.lower_limbs + self.head, 
            "Standing": self.lower_limbs + self.head, 
            "Walking": self.all_limbs, 
            "Waving": self.upper_limbs, 
            "Elbowing": self.upper_limbs, 
            "Frontkicking": self.lower_limbs, 
            "Hamering": self.upper_limbs, 
            "Headering": self.head, 
            "Kneeing": self.lower_limbs, 
            "Pulling": self.upper_limbs, 
            "Punching": self.upper_limbs, 
            "Pushing": self.upper_limbs, 
            "Sidekicking": self.lower_limbs, 
            "Slapping": self.upper_limbs
            }
        self.good_data = {}
        self.normal_actions = ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]
        self.aggressive_actions = ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]
        self.all_actions = self.normal_actions + self.aggressive_actions
        self.best_data = {
            "Bowing": {"l_arm_m2": [1, 2, 6, 7, 8, 9, 10],
                        "l_arm_m3": [1, 2, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 6, 7, 8, 9, 10],
                        "r_arm_m5": [1, 2, 6, 7, 8, 9],
                        "head": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]},
            "Clapping": {"l_arm_m2": [1, 2, 5, 7, 9],
                        "l_arm_m3": [1, 2, 3, 4, 5, 7, 9, 10],
                        "r_arm_m4": [1, 2, 7, 9, 10],
                        "r_arm_m5": [1, 2, 3, 6, 7, 8, 9, 10]}, 
            "Handshaking": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 7, 9, 10],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 7, 9, 10],
                        "r_arm_m4": [1, 2, 3, 4, 7, 9, 10],
                        "r_arm_m5": [1, 2, 3, 4, 7, 9, 10]}, 
            "Hugging": {"l_arm_m2": [1, 5, 6, 8, 9],
                        "l_arm_m3": [1, 5, 6, 8, 9],
                        "r_arm_m4": [2, 3, 4, 8, 9],
                        "r_arm_m5": [2, 3, 4, 8, 9]}, 
            "Jumping": {"l_leg_m6": [1, 2, 3, 4, 5, 8, 9, 10],
                        "l_leg_m7": [1, 3, 4, 5, 8, 9, 10],
                        "r_leg_m8": [1, 3, 4, 5, 8, 9, 10],
                        "r_leg_m9": [3, 4, 8, 9, 10]}, 
            "Running": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "l_leg_m6": [1, 2, 3, 4, 5, 7, 8, 9, 10],
                        "l_leg_m7": [1, 2, 3, 4, 5, 7, 8, 9, 10],
                        "r_leg_m8": [1, 2, 3, 4, 5, 7, 8, 9, 10],
                        "r_leg_m9": [1, 2, 3, 4, 5, 7, 8, 9, 10],
                        "head": [1, 2, 3, 4, 5, 7, 8, 9, 10]}, 
            "Seating": {"l_leg_m6": [1, 3, 4, 5, 6, 7, 8, 9],
                        "l_leg_m7": [1, 3, 5, 6, 7, 9, 10],
                        "r_leg_m8": [3, 4, 5, 6, 8, 9],
                        "r_leg_m9": [3, 4, 5, 6, 7, 9, 10],
                        "head": [2, 3, 4, 5, 10]}, 
            "Standing": {"l_leg_m6": [1, 2, 3, 6, 7, 8, 9],
                        "l_leg_m7": [1, 2, 3, 8, 9],
                        "r_leg_m8": [2, 3, 4, 5, 6, 9],
                        "r_leg_m9": [2, 4, 5, 9],
                        "head": [2, 3, 4, 6, 8]}, 
            "Walking": {"l_arm_m2": [1, 3, 4, 5, 6, 7],
                        "l_arm_m3": [1, 3, 4, 5, 6, 7],
                        "r_arm_m4": [1, 3, 4, 6, 7, 9],
                        "r_arm_m5": [1, 3, 4, 6, 7, 9],
                        "l_leg_m6": [1, 3, 4, 5, 6, 7],
                        "l_leg_m7": [1, 3, 4, 5, 6, 7],
                        "r_leg_m8": [1, 3, 4, 5, 6, 7, 9],
                        "r_leg_m9": [1, 3, 4, 5, 6, 7, 9],
                        "head": [1, 3, 4, 5, 6, 9]}, 
            "Waving": {"l_arm_m2": [1, 2, 5, 7, 9],
                        "l_arm_m3": [1, 2, 5, 7, 8, 9],
                        "r_arm_m4": [1, 2, 5, 7, 9],
                        "r_arm_m5": [1, 2, 3, 7, 9]}, 
            "Elbowing": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m5": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 
            "Frontkicking": {"l_leg_m6": [1, 3, 4, 5, 7, 8, 9],
                            "l_leg_m7": [1, 3, 4, 5, 7, 8, 9],
                            "r_leg_m8": [1, 2, 3, 5, 6, 8, 9, 10],
                            "r_leg_m9": [1, 2, 3, 5, 6, 8, 9, 10],
                            "head": [2, 3, 4, 6, 8]}, 
            "Hamering": {"l_arm_m2": [1, 2, 4, 5, 7, 8, 10],
                        "l_arm_m3": [1, 2, 4, 5, 7, 8, 10],
                        "r_arm_m4": [1, 2, 4, 5, 7, 8, 10],
                        "r_arm_m5": [1, 2, 4, 5, 7, 8, 10]}, 
            "Headering": {"head": [1, 2, 3, 4, 5, 7, 8, 9, 10]}, 
            "Kneeing": {"l_leg_m6": [1, 3, 5, 6, 7, 8, 9],
                        "l_leg_m7": [1, 3, 5, 6, 7, 8, 9],
                        "r_leg_m8": [1, 2, 3, 4, 5, 7, 8, 9],
                        "r_leg_m9": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}, 
            "Pulling": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 4, 6, 8, 9, 10],
                        "r_arm_m5": [1, 2, 3, 5, 6, 8, 9, 10]}, 
            "Punching": {"l_arm_m2": [1, 2, 3, 5, 7, 8, 9],
                        "l_arm_m3": [1, 2, 3, 5, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 4, 5, 6, 7, 8, 9],
                        "r_arm_m5": [1, 2, 4, 5, 6, 7, 8, 9]}, 
            "Pushing": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 9],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 9],
                        "r_arm_m4": [1, 2, 3, 5, 6, 8, 9, 10],
                        "r_arm_m5": [1, 2, 3, 5, 6, 8, 9, 10]}, 
            "Sidekicking": {"l_leg_m6": [2, 3, 4, 5, 6, 7, 8, 9],
                            "l_leg_m7": [2, 3, 4, 5, 6, 7, 8, 9],
                            "r_leg_m8": [1, 2, 3, 5, 7, 8],
                            "r_leg_m9": [1, 2, 3, 5, 7, 8]}, 
            "Slapping": {"l_arm_m2": [1, 2, 3, 4, 5, 6, 7, 8, 9],
                        "l_arm_m3": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                        "r_arm_m4": [1, 2, 3, 4, 5, 6, 8, 9],
                        "r_arm_m5": [1, 2, 3, 4, 5, 8, 9, 10]}}

    def split_by_subact(self, subject, action):
        """
        Splits all normalized data by a subject+action combo

        Args:
            subject --> int; {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
            action ---> string; action name from self.all_actions
        """
        return self.all_data_norm.loc[(self.all_data_norm["subject"] == subject) & (self.all_data_norm["action"] == action)]
    
    def find_max_window_size(self, action, limb):
        self.max_window_size = 1e9
        for sub in self.best_data[action][limb]:
            subact = self.split_by_subact(sub, action)
            window_size = subact.shape[0]
            if window_size < self.max_window_size:
                self.max_window_size = window_size
        self.max_offset = int(0.25*self.max_window_size)

    def locate_best_step(self, sub_list, action, limb):
        """
        - First sub will be stationary
        """
        subact_df_1 = self.split_by_subact(sub_list[0], action)
        subact_df_2 = self.split_by_subact(sub_list[1], action)
        num = 51
        assert num % 2 != 0, f"[FATAL]: Num needs to be odd! Got num={num}. Check Analyzer.locate_best_step()."
        steps = np.linspace(-self.max_offset, self.max_offset, num).astype(int)
        errors = np.empty((num, 2))
        for i in range(num):
            self.step = steps[i]
            error = self.calc_error(subact_df_1[limb + "_c"].iloc[:self.max_window_size], subact_df_2[limb + "_c"].iloc[:self.max_window_size])
            errors[i, :] = [error, self.step]
        best_error = np.argmin(errors[:, 0])
        best_offset = errors[best_error, 1]
        return best_offset

    def calc_error(self, subact_col_1, subact_col_2):
        subact_col_1_np = subact_col_1.to_numpy()
        subact_col_2_np = subact_col_2.to_numpy()
        if self.step < 0:
            error = np.sqrt(np.sum(np.square(np.subtract(subact_col_1_np[:self.max_window_size + self.step], subact_col_2_np[-self.step:self.max_window_size]))))/(self.max_window_size+self.step)
        else:
            error = np.sqrt(np.sum(np.square(np.subtract(subact_col_1_np[self.step:self.max_window_size], subact_col_2_np[:self.max_window_size - self.step]))))/(self.max_window_size-self.step)
        return error

    def calc_shapelet_error(self, test, true, step):
        window_size = min(test.shape[0], true.shape[0])
        if step < 0:
            error = np.sqrt(np.sum(np.square(np.subtract(test[:window_size + step], true[-step:window_size]))))/(window_size+step)
        else:
            error = np.sqrt(np.sum(np.square(np.subtract(test[step:window_size], true[:window_size - step]))))/(window_size-step)
        return error

    def find_offsets(self, action, limb):
        num_limbs = len(list(self.best_data[action][limb][1:]))
        self.offsets = np.empty((num_limbs, 2))
        i = 0
        for sub in self.best_data[action][limb][1:]:
            best_offset = self.locate_best_step([self.best_data[action][limb][0], sub], action, limb)
            self.offsets[i, :] = [sub, best_offset]
            i += 1

    def find_offsets_TRAIN(self, action, limb):
        num_limbs = len(list(self.train_data[action][limb][1:]))
        self.offsets = np.empty((num_limbs, 2))
        i = 0
        for sub in self.train_data[action][limb][1:]:
            best_offset = self.locate_best_step([self.train_data[action][limb][0], sub], action, limb)
            self.offsets[i, :] = [sub, best_offset]
            i += 1

    def plot_shapelet(self, action, limb, type):
        if type == "all":
            num_subs = len(self.best_data[action][limb])
            if num_subs % 2 != 0:
                subs = self.best_data[action][limb][:-1]
            else:
                subs = self.best_data[action][limb]
            num_plots = len(subs)
            assert num_plots % 2 == 0, "Ensure the number of plots is even for plotting reasons!"
            fig, axs = plt.subplots(2, int(num_plots/2))
            x = 0
            y = 0
            for sub in subs:
                subact = self.split_by_subact(sub, action)
                axs[y, x].plot(subact["timestamp"].iloc[:self.max_window_size], subact[limb + "_c"].iloc[:self.max_window_size], c='b')
                axs[y, x].plot(range(self.shapelet.shape[0]), self.shapelet, c='r')
                axs[y, x].set_title(f"Sub: {sub}")
                if x == int(num_plots/2-1):
                    y += 1
                    x = 0
                else:
                    x += 1
            fig.suptitle(f"Shapelet comparison for A: {action}, L: {limb}")
            plt.show()
        elif type == "train":
            num_subs = len(self.test_data[action][limb])
            if num_subs % 2 != 0:
                subs = self.test_data[action][limb][:-1]
            else:
                subs = self.test_data[action][limb]
            num_plots = len(subs)
            assert num_plots % 2 == 0, "Ensure the number of plots is even for plotting reasons!"
            fig, axs = plt.subplots(2, int(num_plots/2))
            x = 0
            y = 0
            for sub in subs:
                subact = self.split_by_subact(sub, action)
                axs[y].plot(subact["timestamp"].iloc[:self.max_window_size], subact[limb + "_c"].iloc[:self.max_window_size], c='b')
                axs[y].plot(range(self.shapelet.shape[0]), self.shapelet, c='r')
                axs[y].set_title(f"Sub: {sub}")
                if x == int(num_plots/2-1):
                    y += 1
                    x = 0
                else:
                    x += 1
            fig.suptitle(f"Shapelet comparison for A: {action}, L: {limb}")
            plt.show()

    def create_shapelets_by_avg(self, action, limb):
        self.find_max_window_size(action, limb)                                     # Returns self.max_window_size, self.max_offset
        self.find_offsets(action, limb)                                             # Returns self.offsets
        subact_0 = self.split_by_subact(self.best_data[action][limb][0], action)
        shapelet = subact_0[limb+"_c"].iloc[:self.max_window_size].to_numpy()       # Init where as self.best_data[action][limb][0]
        shapelet_count = np.ones((self.max_window_size))                            # Init to ones
        for row in range(self.offsets.shape[0]):
            sub = self.offsets[row, 0]
            offset = int(self.offsets[row, 1])
            subact = self.split_by_subact(sub, action)
            new = subact[limb+"_c"].iloc[:self.max_window_size].to_numpy()
            if offset < 0:
                shapelet[:self.max_window_size+offset] += new[-offset:]
                shapelet_count[:self.max_window_size+offset] += 1
            else:
                shapelet[offset:] += new[:self.max_window_size-offset]
                shapelet_count[offset:] += 1
        self.shapelet = shapelet/shapelet_count
    
    def split_data(self):
        self.train_data = {}
        self.test_data = {}
        for action in list(self.best_data.keys()):
            self.train_data.update({action: {}})
            self.test_data.update({action: {}})
            for limb in list(self.best_data[action].keys()):
                random.shuffle(self.best_data[action][limb])
                num_limb = len(self.best_data[action][limb])
                if num_limb in [4, 5, 6]:
                    num_test = 1 #2
                elif num_limb in [7, 8]:
                    num_test = 1 #3
                elif num_limb in [9, 10]:
                    num_test = 1 #4
                self.test_data[action].update({limb: self.best_data[action][limb][:num_test]})
                self.train_data[action].update({limb: self.best_data[action][limb][num_test:]})
    
    def create_shapelets_by_avg_TRAIN(self, action, limb):
        action = random.choice(list(self.best_data.keys()))
        limb = random.choice(list(self.best_data[action].keys()))
        self.find_max_window_size(action, limb)                                     # Returns self.max_window_size, self.max_offset
        self.find_offsets_TRAIN(action, limb)                                             # Returns self.offsets
        subact_0 = self.split_by_subact(self.train_data[action][limb][0], action)
        shapelet = subact_0[limb+"_c"].iloc[:self.max_window_size].to_numpy()       # Init where as self.best_data[action][limb][0]
        shapelet_count = np.ones((self.max_window_size))                            # Init to ones
        for row in range(self.offsets.shape[0]):
            sub = self.offsets[row, 0]
            offset = int(self.offsets[row, 1])
            subact = self.split_by_subact(sub, action)
            new = subact[limb+"_c"].iloc[:self.max_window_size].to_numpy()
            if offset < 0:
                shapelet[:self.max_window_size+offset] += new[-offset:]
                shapelet_count[:self.max_window_size+offset] += 1
            else:
                shapelet[offset:] += new[:self.max_window_size-offset]
                shapelet_count[offset:] += 1
        self.shapelet = shapelet/shapelet_count

    def create_ALL_shapelets(self):
        self.counter = 1
        self.all_shapelets = {}
        for action in list(self.best_data.keys()):
            self.all_shapelets.update({action: {}})
            for limb in list(self.best_data[action].keys()):
                self.create_shapelets_by_avg(action, limb)
                self.all_shapelets[action].update({limb: self.shapelet})
                self.counter += 1

    def create_shapelets_TRAIN(self):
        self.counter = 1
        self.train_shapelets = {}
        for action in list(self.train_data.keys()):
            self.train_shapelets.update({action: {}})
            for limb in list(self.train_data[action].keys()):
                self.create_shapelets_by_avg_TRAIN(action, limb)
                self.train_shapelets[action].update({limb: self.shapelet})
                self.counter += 1

    def compare_shapelet(self, test_shapelet, true_shapelet):
        num = 101
        max_offset = int(true_shapelet.shape[0]/4)
        assert num % 2 != 0, f"[FATAL]: Num needs to be odd! Got num={num}. Check Analyzer.compare_shapelet()."
        steps = np.linspace(-max_offset, max_offset, num).astype(int)
        errors = np.empty((num, 2))
        for i in range(num):
            step = steps[i]
            error = self.calc_shapelet_error(test_shapelet, true_shapelet, step)
            errors[i, :] = [error, step]
        errors_sort = np.sort(errors[:, 0])
        avg_error = np.mean(errors_sort[:5])
        best_error_loc = np.argmin(errors[:, 0])
        best_error = errors[best_error_loc, 0]
        best_offset = errors[best_error_loc, 1]
        return best_error

    def calc_shapelet_error_SHIFT(self, test, true, step, inc):
        if test[step:step+inc].shape[0] != true.shape[0]:
            error = np.sqrt(np.sum(np.square(np.subtract(test[step-1:step+inc], true))))
        else:
            error = np.sqrt(np.sum(np.square(np.subtract(test[step:step+inc], true))))
        return error

    def compare_shapelet_SHIFT(self, test_shapelet, true_shapelet):
        num = 101
        inc = int(true_shapelet.shape[0]/3)
        # 1st Third
        true_part = true_shapelet[:inc]
        # # Middle Third
        # true_part = true_shapelet[inc:2*inc]
        # # Last 3rd
        # true_part = true_shapelet[2*inc:]
        steps = np.linspace(0, 2*inc, num).astype(int)
        errors = np.empty((num, 2))
        for i in range(num):
            step = steps[i]
            error = self.calc_shapelet_error_SHIFT(test_shapelet, true_part, step, inc)
            errors[i, :] = [error, step]
        errors_sort = np.sort(errors[:, 0])
        avg_error = np.mean(errors_sort[:10])
        best_error_loc = np.argmin(errors[:, 0])
        best_error = errors[best_error_loc, 0]
        best_offset = errors[best_error_loc, 1]
        return avg_error

    def initialize_results(self):
        self.results_master = {}
        for action in list(self.best_data.keys()):
            self.results_master.update({action: {}})
            for limb in list(self.best_data[action].keys()):
                self.results_master[action].update({limb: np.zeros((10, 2))})

    def run(self, mode):
        num_trials = 300
        self.initialize_results()
        if mode == "all":
            self.create_ALL_shapelets()
        for _ in range(num_trials):
            trial_start = time.time()
            print(f"[INFO]: Running Trial {_+1} / {num_trials}...\t\t{round(((_+1)/num_trials)*100, 2)}%")
            self.split_data()

            if mode == "train":
                self.create_shapelets_TRAIN()

            ####################################################
            ##### Test all Action+Limb combos in test_data #####
            ####################################################
            # print(f"Action\t\tAccuracy")
            for a1 in list(self.test_data.keys()):
                total = 0
                correct = 0
                for limb in list(self.test_data[a1].keys()):
                    test_sub = self.test_data[a1][limb][0]
                    subact_df = self.split_by_subact(test_sub, a1)
                    test_shapelet = subact_df[limb + "_c"].to_numpy()
                    min_error = 1000

                    ################################################################
                    ##### Compare each combo to all shapelets of the same limb #####
                    ################################################################
                    if mode == "train":
                        for a2 in list(self.train_shapelets.keys()):
                            if limb not in list(self.train_shapelets[a2].keys()):
                                continue
                            else:
                                true_shapelet = self.train_shapelets[a2][limb]
                                max_length = min(test_shapelet.shape[0], true_shapelet.shape[0])
                                error = self.compare_shapelet_SHIFT(test_shapelet[:max_length], true_shapelet[:max_length])
                                if error < min_error:
                                    action_pred = a2
                                    min_error = error
                        total += 1
                        self.results_master[a1][limb][test_sub-1, :] += np.array([0, 1])
                        if action_pred == a1:
                            correct += 1
                            self.results_master[a1][limb][test_sub-1, :] += np.array([1, 0])
                        # print(f"L: {limb}\tPred: {action_pred}")
                    elif mode == "all":
                        for a2 in list(self.all_shapelets.keys()):
                            if limb not in list(self.all_shapelets[a2].keys()):
                                continue
                            else:
                                true_shapelet = self.all_shapelets[a2][limb]
                                max_length = min(test_shapelet.shape[0], true_shapelet.shape[0])
                                # error = self.compare_shapelet(test_shapelet[:max_length], true_shapelet[:max_length])
                                error = self.compare_shapelet_SHIFT(test_shapelet[:max_length], true_shapelet[:max_length])
                                if error < min_error:
                                    action_pred = a2
                                    min_error = error
                        total += 1
                        self.results_master[a1][limb][test_sub-1, :] += np.array([0, 1])
                        if action_pred == a1:
                            correct += 1
                            self.results_master[a1][limb][test_sub-1, :] += np.array([1, 0])
                        # print(f"L: {limb}\tPred: {action_pred}")
                # print(f"{a1}\t\t\t{round(correct/total*100, 2)}%")
            print(f"[INFO]: Trial {_+1} took {round(time.time() - trial_start, 2)} s")
    
    def run_RMS(self):
        num_trials = 1
        self.initialize_results()
        errors = {}
        errors_master = np.zeros((20, 2))
        for _ in range(num_trials):
            trial_start = time.time()
            print(f"[INFO]: Running Trial {_+1} / {num_trials}...\t\t{round(((_+1)/num_trials)*100, 2)}%")
            self.split_data()

            ####################################################
            ##### Test all Action+Limb combos in test_data #####
            ####################################################
            # for a1 in list(self.test_data.keys()):
            for a1 in ["Clapping"]:
                errors = {}
                # for limb in list(self.test_data[a1].keys()):
                for limb in ["l_arm_m3"]:
                    name = a1 + "_" + limb
                    errors.update({name: np.empty((0, 2))})
                    test_sub = self.test_data[a1][limb][0]
                    subact_df = self.split_by_subact(test_sub, a1)
                    test_shapelet = subact_df[limb + "_c"].to_numpy()

                    ################################################################
                    ##### Compare each combo to all shapelets of the same limb #####
                    ################################################################

                    action_num = 0
                    for a2 in list(self.train_data.keys()):
                        if limb not in list(self.train_data[a2].keys()):
                            continue
                        else:
                            # for limb_2 in list(self.train_data[a2].keys()):
                            for sub in self.train_data[a2][limb]:
                                true_subact_df = self.split_by_subact(sub, a2)
                                true_shapelet = true_subact_df[limb + "_c"].to_numpy()
                                max_length = min(test_shapelet.shape[0], true_shapelet.shape[0])
                                min_error = self.compare_shapelet(test_shapelet[:max_length], true_shapelet[:max_length])
                                #store min error
                                errors[name] = np.concatenate((errors[name], np.array([[min_error, int(action_num)]])), axis=0)
                            action_num += 1
                    min_errors = np.sort(errors[name][:, 0])
                    actions = []
                    for er in min_errors[:8]:
                        loc, = np.where(errors[name][:, 0] == er)
                        actions.append(errors[name][loc[0], 1])
                    action_pred = self.all_actions[int(statistics.mode(actions))]
                    if action_pred == a1:
                        errors_master[action_num, :] += np.array([1, 1])
                    else:
                        errors_master[action_num, :] += np.array([0, 1])
            
        #     print(f"[INFO]: Trial {_+1} took {round(time.time() - trial_start, 2)} s")
        # print(errors_master)
        # print(errors)
        


# Read all_data_norm df
print("[TRACE]: Beginning read process ...")
start_read = time.time()
print("[INFO]: Reading all_data_norm.csv ...")
all_data_norm = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_norm.csv")
print(f"[TRACE]: Done with reading process. Took {round(time.time() - start_read, 2)} s")

# Analyze class
anlyz = Analyzer(all_data_norm)

# Run
# anlyz.locate_best_step([5, 6], "Jumping", "r_leg", 8, "c", 25)
# for _ in range(10):
# anlyz.create_shapelets_by_avg("Hugging", "l_arm_m3")
# anlyz.create_shapelets_by_avg_TRAIN("Hugging", "l_arm_m3")
# anlyz.create_ALL_shapelets()
# anlyz.run("all")
# print(anlyz.results_master)

# anlyz.split_data()
# action = random.choice(anlyz.all_actions)
# limb = random.choice(list(anlyz.best_data[action].keys()))
# anlyz.create_shapelets_by_avg(action, limb)
# anlyz.plot_shapelet(action, limb, "all")

# anlyz.run_RMS()