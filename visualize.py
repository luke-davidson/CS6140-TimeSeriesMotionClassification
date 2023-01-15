from telnetlib import X3PAD
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
from scipy import stats

class Visualizer():
    """
    
    """
    def __init__(self, all_data, all_data_norm):
        self.column_names = ["subject", "action_type", "action", 
                            "timestamp", 
                            "head_x", "head_y", "head_z", 
                            "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", 
                            "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", 
                            "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", 
                            "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", 
                            "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", 
                            "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", 
                            "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", 
                            "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z"]
        self.all_data = all_data
        self.all_data_norm = all_data_norm
        self.normal_actions = ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]
        self.aggressive_actions = ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]
        self.all_actions = self.normal_actions + self.aggressive_actions
        self.limbs = ["head", "l_arm", "r_arm", "l_leg", "r_leg"]
        self.limb_col = ["head_x", "head_y", "head_z", 
                            "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", 
                            "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", 
                            "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", 
                            "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", 
                            "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", 
                            "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", 
                            "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", 
                            "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z"]
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

    def split_by_subact(self, df, subject, action):
        """
        Splits all_data by a given subject and action, returns DataFrame of the subject+action data

        Args:
            df -------> string; "all" or "norm", specifying which df to read from
            subject --> int; 1-10
            action ---> string; from self.all_actions
        """
        if df == "all":
            return self.all_data.loc[(self.all_data["subject"] == subject) & (self.all_data["action"] == action)]
        elif df == "norm":
            return self.all_data_norm.loc[(self.all_data_norm["subject"] == subject) & (self.all_data_norm["action"] == action)]
        else:
            raise Exception(f"Invalid df input of \"{df}\". Must be \"all\" or \"norm\".")
    
    def plot_3D(self, subact_df, limb, sensor):
        """
        Plots 3D movement of a certain limb from a certain subject+action combo

        Args:
            subact_df --> pd.DataFrame; subject+action data likely from self.split_by_subact
            limb -------> string; {"head", "l_arm", "r_arm", "l_leg", "r_leg"}
            sensor -----> int; 2-9 corresponding to the sensor to use
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        if limb == "head":
            x = subact_df[limb + "_x"]
            y = subact_df[limb + "_y"]
            z = subact_df[limb + "_z"]
        else:
            x = subact_df[limb + "_m" + str(sensor) + "_x"]
            y = subact_df[limb + "_m" + str(sensor) + "_y"]
            z = subact_df[limb + "_m" + str(sensor) + "_z"]
        ax.plot(x, y, z, 'gray')
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.show()

    def plot_2D(self, df_mode, subject, action, limb, sensor, coord):
        """
        Plots 2D time series of a certain limb+coordinate of a certain subject+action combo

        Args:
            subact_df --> pd.DataFrame; subject+action data likely from self.split_by_subact
            limb -------> string; {"head", "l_arm", "r_arm", "l_leg", "r_leg"}
            sensor -----> int; 2-9 corresponding to the sensor to use
            coord ------> string; {"x", "y", "z", "c"}
        """
        subact_df = self.split_by_subact(df_mode, subject, action)
        if coord == "c":
            if limb == "head":
                plt.plot(subact_df["timestamp"], subact_df[limb + "_c"])
            else:
                plt.plot(subact_df["timestamp"], subact_df[limb + "_m" + str(sensor) + "_c"])
        else:
            if limb == "head":
                plt.plot(subact_df["timestamp"], subact_df[limb + "_" + coord])
            else:
                plt.plot(subact_df["timestamp"], subact_df[limb + "_m" + str(sensor) + "_" + coord])
        subject = subact_df["subject"].iloc[0]
        action = subact_df["action"].iloc[0]
        plt.title(f"Timeseries for S: {subject}, A: {action}, L: {limb} m{sensor}")
        plt.xlabel("Time")
        plt.ylabel("Position")
        plt.show()

    def plot_2D_both(self, subject, action, limb, sensor, coord):
        subact_all_df = self.split_by_subact("all", subject, action)
        subact_norm_df = self.split_by_subact("norm", subject, action)
        if coord == "c":
            if limb == "head":
                y_all = subact_all_df[limb + "_c"]
                y_norm = subact_norm_df[limb + "_c"]
            else:
                y_all = subact_all_df[limb + "_m" + str(sensor) + "_c"]
                y_norm = subact_norm_df[limb + "_m" + str(sensor) + "_c"]
        else:
            if limb == "head":
                y_all = subact_all_df[limb + "_" + coord]
                y_norm = subact_norm_df[limb + "_" + coord]
            else:
                y_all = subact_all_df[limb + "_m" + str(sensor) + "_" + coord]
                y_norm = subact_norm_df[limb + "_m" + str(sensor) + "_" + coord]
        x_all = subact_all_df["timestamp"]
        x_norm = subact_norm_df["timestamp"]
        subject = subact_all_df["subject"].iloc[0]
        action = subact_all_df["action"].iloc[0]
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.suptitle(f"Raw and Normalized Timeseries for S: {subject}, A: {action}, L: {limb} m{sensor}")
        ax1.plot(x_all, y_all)
        ax1.set_title("All Data")
        ax2.plot(x_norm, y_norm)
        ax2.set_title("Normalized Data")
        plt.show()
    
    def plot_subacts(self, sub_list, df_mode, action, limb, sensor, coord):
        """
        
        """
        assert len(sub_list) % 2 == 0, "Ensure the length of the list of subjects is even for plotting reasons!"
        fig, axs = plt.subplots(2, int(len(sub_list)/2))
        xs = range(int(len(sub_list)/2))
        x = 0
        y = 0
        for sub in sub_list:
            subact_df = self.split_by_subact(df_mode, sub, action)
            if limb == "head":
                axs[y, x].plot(subact_df["timestamp"], subact_df[limb + "_" + coord])
                axs[y,x].set_title(f"S: {sub}")
            else:
                axs[y, x].plot(subact_df["timestamp"], subact_df[limb + "_m" + str(sensor) + "_" + coord])
                axs[y,x].set_title(f"S: {sub}")
            x += 1
            if x == len(sub_list)/2:
                y += 1
                x = 0
        fig.suptitle(f"Data for A: {action}, L: {limb} m{sensor}")
        plt.show()
    
    def plot_offsets(self, sub_list, df_mode, action, limb, sensor, coord, inc):
        """
        - First sub will be stationary
        - Make num_plots divis by 4
        - Some plots
            # 1 and 5 for Frontkicking r_leg_8
            # 1, 2, 5, 7, 9 Clapping r_arm_2
        """
        num_plots = 12
        assert num_plots % 4 == 0, "Ensure the number of plots is divisible by 4 for plotting reasons!"
        subact_df_1 = self.split_by_subact(df_mode, sub_list[0], action)
        subact_df_2 = self.split_by_subact(df_mode, sub_list[1], action)
        self.max_window_size = min(subact_df_1.shape[0], subact_df_2.shape[0])
        fig, axs = plt.subplots(int(num_plots/4), 4)
        x = 0
        y = 0
        self.step = -500
        for _ in range(num_plots):
            if limb == "head":
                axs[y, x].plot(subact_df_1["timestamp"].iloc[:self.max_window_size], subact_df_1[limb + "_" + coord].iloc[:self.max_window_size])
                axs[y, x].plot(subact_df_2["timestamp"].iloc[:self.max_window_size] + self.step, subact_df_2[limb + "_" + coord].iloc[:self.max_window_size])
                axs[y, x].set_title(f"Step: {self.step}")
            else:
                axs[y, x].plot(subact_df_1["timestamp"].iloc[:self.max_window_size], subact_df_1[limb + "_m" + str(sensor) + "_" + coord].iloc[:self.max_window_size])
                axs[y, x].plot(subact_df_2["timestamp"].iloc[:self.max_window_size] + self.step, subact_df_2[limb + "_m" + str(sensor) + "_" + coord].iloc[:self.max_window_size])
                axs[y, x].set_title(f"Step: {self.step}")
            # Plotting axes
            if x == 3:
                y += 1
                x = 0
            else:
                x += 1
            self.step += inc
        fig.suptitle(f"Data for A: {action}, L: {limb} m{sensor}")
        plt.show()



# Read all_data dfs
print("[TRACE]: Beginning read process ...")
start_read = time.time()
print("[INFO]: Reading all_data.csv ...")
# all_data = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data.csv")
all_data = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_WITH_ZEROS.csv")
print("[INFO]: Reading all_data_norm.csv ...")
all_data_norm = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_norm.csv")
print(f"[TRACE]: Done with reading process. Took {round(time.time() - start_read, 2)} s")

# Viz class
viz = Visualizer(all_data, all_data_norm)

# subact = viz.split_by_subact("all", 3, "Elbowing")
# viz.plot_3D(subact, "r_arm", 4)
# viz.plot_subacts(range(1, 11, 1), "norm", "Clapping", "r_arm", 5, "c")
max = 1000
inc = int(max/12)
# viz.plot_offsets([1, 5], "norm", "Frontkicking", "r_leg", 8, "c", inc)
# viz.plot_offsets(sub_list, df_mode, action, limb, sensor, coord, inc)