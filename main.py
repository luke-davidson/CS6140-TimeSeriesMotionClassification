from telnetlib import X3PAD
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import random
import time
from scipy import stats

class Analyzer():
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
        self.sensor_num_per_limb = {}
        # for 

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
                





# Read all_data dfs
print("[TRACE]: Beginning read process ...")
start_read = time.time()
print("[INFO]: Reading all_data.csv ...")
all_data = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data.csv")
print("[INFO]: Reading all_data_norm.csv ...")
all_data_norm = pd.read_csv("/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_norm.csv")
print(f"[TRACE]: Done with reading process. Took {round(time.time() - start_read, 2)} s")




# Create class
anlyz = Analyzer(all_data, all_data_norm)
# anlyz.plot_2D("all", 4, "Punching", "l_arm", 3, "c")
# anlyz.plot_2D_both(4, "Clapping", "l_arm", 2, "c")
anlyz.plot_subacts(range(1, 11, 1), 
                    "norm", 
                    "Elbowing", 
                    "r_arm", 
                    4, 
                    "c")

"""
["head_x", "head_y", "head_z", "head_c", 
"l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", "l_arm_m2_c", 
"l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", "l_arm_m3_c", 
"r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", "r_arm_m4_c", 
"r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", "r_arm_m5_c", 
"l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", "l_leg_m6_c", 
"l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", "l_leg_m7_c", 
"r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", "r_leg_m8_c", 
"r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z", "r_leg_m9_c"]

"""