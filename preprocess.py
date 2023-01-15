import numpy as np
import pandas as pd
import time

class PreProcess():
    """
    Pre-processing class for human motion time series data classification
    """
    def __init__(self, dataset_file_path):
        self.master_file_path = dataset_file_path
        self.normal_actions = ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]
        self.aggressive_actions = ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]
        self.all_actions = self.normal_actions + self.aggressive_actions
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
        self.column_names_short = ["timestamp", 
                            "head_x", "head_y", "head_z", 
                            "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", 
                            "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", 
                            "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", 
                            "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", 
                            "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", 
                            "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", 
                            "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", 
                            "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z"]
        self.limb_col = ["head_x", "head_y", "head_z", "head_c", 
                            "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", "l_arm_m2_c", 
                            "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", "l_arm_m3_c", 
                            "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", "r_arm_m4_c",
                            "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", "r_arm_m5_c",
                            "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", "l_leg_m6_c",
                            "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", "l_leg_m7_c",
                            "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", "r_leg_m8_c",
                            "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z", "r_leg_m9_c"]
        self.limb_col_combo = [("head_", 7), ("l_arm_m2_", 11), ("l_arm_m3_", 15), ("r_arm_m4_", 19), 
                            ("r_arm_m5_", 23), ("l_leg_m6_", 27), ("l_leg_m7_", 31), ("r_leg_m8_", 35), ("r_leg_m9_", 39)]
        self.all_data_df = pd.DataFrame(columns=self.column_names)
        self.save_all_filepath = "/Users/lukedavidson/Documents/CS6140_FinalProj/all_data.csv"
        self.save_norm_filepath = "/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_norm.csv"

    def read(self):
        """
        Read all individual files for each subject and store them in a pd.df
        """
        print("[TRACE]: Beginning read process...")
        count = 0
        total_iters = 200
        start_time = time.time()
        for n in range(1, 11):
            for type in ["aggressive", "normal"]:
                folder_name = "sub" + str(n) + "/" + type + "/"
                if type == "aggressive":
                    for action in self.aggressive_actions:
                        file_path = self.master_file_path + folder_name + action + ".txt"
                        df = pd.read_csv(file_path, sep = '\s+', names = self.column_names_short)
                        # df["timestamp"] = pd.Series(range(df.shape[0]))
                        df.insert(0, "subject", n)
                        df.insert(1, "action_type", type)
                        df.insert(2, "action", action)
                        df = df.drop(columns=["timestamp"])
                        df.insert(3, "timestamp", pd.Series(range(df.shape[0])))
                        df = df.drop(index=[0])
                        self.all_data_df = pd.concat([self.all_data_df, df], axis=0)
                        count += 1
                        print(f"[INFO]: Done reading file \t{folder_name}{action}.txt \t\t Read {round(count/total_iters*100, 2)}% complete")
                else:
                    # type == "normal"
                    for action in self.normal_actions:
                        if n == 10 and action == "Waving":
                            # for some reason no data for sub10/normal/Waving
                            pass
                        else:
                            file_path = self.master_file_path + folder_name + action + ".txt"
                            df = pd.read_csv(file_path, sep = '\s+', names = self.column_names_short)
                            # df["timestamp"] = pd.Series(range(df.shape[0]))
                            df.insert(0, "subject", n)
                            df.insert(1, "action_type", type)
                            df.insert(2, "action", action)
                            df = df.drop(columns=["timestamp"])
                            df.insert(3, "timestamp", pd.Series(range(df.shape[0])))
                            df = df.drop(index=[0])
                            self.all_data_df = pd.concat([self.all_data_df, df], axis=0)
                            count += 1
                            print(f"[INFO]: Done reading file \t{folder_name}{action}.txt \t\t Read {round(count/total_iters*100, 2)}% complete")
        print("[TRACE]: Done with reading process.")
        print(f"[INFO]: Reading process took {round(time.time() - start_time, 2)} s")
    
    def check_for_zeros(self):
        """
        Checks whether there are zero values in the data, which are essentially errors in data recording
        """
        self.all_data_np = self.all_data_df.to_numpy()
        self.loc_y, self.loc_x = np.where(self.all_data_np == 0)
        if self.loc_y.shape[0] > 0:
            print(f"[WARN]: Found {self.loc_y.shape[0]} zeros!")
            foundZeros = True
        else:
            foundZeros = False
        return foundZeros

    def replace_zeros(self):
        """
        Replaces the zeros in the data, which ultimately mess up scaling. Zeros in the sensor data means that 
            there was an error in the sensor reading, or maybe different data frequencies, etc. I replace them 
            by setting them equal to the closest last non-0 value.
        """
        zero_count = 0
        print("[TRACE]: Beginning replacing zeros ...")
        for y, x in zip(self.loc_y, self.loc_x):
            top = False
            bottom = False
            inc = 1
            if y <= self.all_data_np.shape[0]/2:
                # if y is in top half, go down
                top = True
                while self.all_data_np[np.clip(y+inc, 0, self.all_data_np.shape[0]-1), x] == 0:
                    inc += 1
            else:
                # if y is in bottom half, go up
                bottom = True
                while self.all_data_np[np.clip(y-inc, 0, self.all_data_np.shape[0]-1), x] == 0:
                    inc += 1
            if top:
                self.all_data_np[y, x] = self.all_data_np[np.clip(y+inc, 0, self.all_data_np.shape[0]-1), x]
            elif bottom:
                self.all_data_np[y, x] = self.all_data_np[np.clip(y-inc, 0, self.all_data_np.shape[0]-1), x]
            zero_count += 1
            if zero_count % 1000 == 0:
                print(f"[INFO]: Done replacing {zero_count}/{self.loc_y.shape[0]}\t\t {round(zero_count/self.loc_y.shape[0]*100, 2)}% complete")
        print(f"[INFO]: Done replacing {zero_count}/{self.loc_y.shape[0]}\t\t {round(zero_count/self.loc_y.shape[0]*100, 2)}% complete")
        print("[TRACE]: Done with replacing zeros")
        self.all_data_df = pd.DataFrame(self.all_data_np, columns=self.column_names)

    def combine_xyz(self):
        """
        Combines x, y, and z coordinates using sqrt of sum of squares 
            combination = sqrt(x^2 + y^2 + z^2)
        """
        print("[TRACE]: Beginning xyz coordinate combination calculations and inserts ...")
        for limb, loc in self.limb_col_combo:
            self.all_data_df.insert(loc, limb + "c", (self.all_data_df[limb + "x"]**2 + self.all_data_df[limb + "y"]**2 + self.all_data_df[limb + "z"]**2)**0.5)
            print(f"[INFO]: Done creating combination for {limb}")
        print("[TRACE]: Done with coordinate combination process.")

    def save(self, all_norm):
        """
        Writes the dataframe to the output file location as a .csv file

        Args:
            all_norm --> string, {"all", "norm"}; identifying whether to save all_data_df or all_data_norm_df
        """
        if all_norm == "all":
            print(f"[TRACE]: Saving all_data df to file path {self.save_all_filepath} ...")
            self.all_data_df.to_csv(self.save_all_filepath)
            print("[TRACE]: File successfully saved.")
        elif all_norm == "norm":
            print(f"[TRACE]: Saving all_data_norm df to file path {self.save_norm_filepath} ...")
            self.all_data_norm_df.to_csv(self.save_norm_filepath)
            print("[TRACE]: File successfully saved.")
    
    def normalize(self):
        """
        Normalizes data to the range [0, 1] by shifting each individual subject/actions so that the min is 0, 
            then dividing by the max value
        """
        self.all_data_norm_df = self.all_data_df.copy()
        print("[TRACE]: Beginning normalization process...")
        count = 0
        total_iters = 200
        start_time = time.time()
        for sub in range(1, 11):
            print("\n*********************************")
            print(f"***** Normalizing subject {sub} *****")
            print("*********************************")
            for act in self.all_actions:
                subact = self.all_data_df.loc[(self.all_data_df["subject"] == sub) & (self.all_data_df["action"] == act)]
                for col in self.limb_col:
                    subact[col] += -subact[col].min()
                    denom = subact[col].max()
                    self.all_data_norm_df.loc[(self.all_data_norm_df["subject"] == sub) & (self.all_data_norm_df["action"] == act), col] = subact[col]/denom
                count += 1
                print(f"[INFO]: Done normalizing Subject: {sub}\t Action: {act} \t\tNorm {round(count/total_iters*100, 2)}% complete")
        print(f"[TRACE]: Done with normalization process.")
        print(f"[INFO]: Normalizing process took {round(time.time() - start_time, 2)} s")
    
    def execute(self):
        """
        Execute all methods. 
        Whole process: read --> replace zeros --> combine xyz coords --> save all --> normalize --> save normalized
        """
        print("[TRACE]: Beginning pre-processing ...")
        master_start_time = time.time()
        self.read()
        zeros = self.check_for_zeros()
        if zeros:
            self.replace_zeros()
        self.combine_xyz()
        self.save("all")
        # self.normalize()
        # self.save("norm")
        print("[TRACE]: Done pre-processing")
        print(f"[INFO]: Whole pre-processing stage took {round(time.time() - master_start_time, 2)} s")

# Run all
master_path = "/Users/lukedavidson/Downloads/Vicon_Physical_Action_Data_Set/"
pre = PreProcess(master_path)
pre.execute()