import numpy as np
import pandas as pd
import time

column_names = ["subject", "action_type", "action", 
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

column_names_short = ["timestamp", 
                      "head_x", "head_y", "head_z", 
                      "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", 
                      "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", 
                      "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", 
                      "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", 
                      "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", 
                      "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", 
                      "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", 
                      "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z"]

limb_col = ["head_x", "head_y", "head_z", "head_c", 
            "l_arm_m2_x", "l_arm_m2_y", "l_arm_m2_z", "l_arm_m2_c", 
            "l_arm_m3_x", "l_arm_m3_y", "l_arm_m3_z", "l_arm_m3_c", 
            "r_arm_m4_x", "r_arm_m4_y", "r_arm_m4_z", "r_arm_m4_c",
            "r_arm_m5_x", "r_arm_m5_y", "r_arm_m5_z", "r_arm_m5_c",
            "l_leg_m6_x", "l_leg_m6_y", "l_leg_m6_z", "l_leg_m6_c",
            "l_leg_m7_x", "l_leg_m7_y", "l_leg_m7_z", "l_leg_m7_c",
            "r_leg_m8_x", "r_leg_m8_y", "r_leg_m8_z", "r_leg_m8_c",
            "r_leg_m9_x", "r_leg_m9_y", "r_leg_m9_z", "r_leg_m9_c"]

limb_col_combo = [("head_", 7), ("l_arm_m2_", 11), ("l_arm_m3_", 15), ("r_arm_m4_", 19), 
                  ("r_arm_m5_", 23), ("l_leg_m6_", 27), ("l_leg_m7_", 31), ("r_leg_m8_", 35), ("r_leg_m9_", 39)]

normal_actions = ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]
aggressive_actions = ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]
all_actions = normal_actions + aggressive_actions

all_data_df = pd.DataFrame(columns=column_names)

# Read data to all_data_df
print("[TRACE]: Beginning read process...")
count = 0
total_iters = 200
start_time = time.time()
for n in range(1, 11):
    for type in ["aggressive", "normal"]:
        folder_name = "sub" + str(n) + "/" + type + "/"
        if type == "aggressive":
            for action in ["Elbowing", "Frontkicking", "Hamering", "Headering", "Kneeing", "Pulling", "Punching", "Pushing", "Sidekicking", "Slapping"]:
                file_path = "/Users/lukedavidson/Downloads/Vicon_Physical_Action_Data_Set/" + folder_name + action + ".txt"
                df = pd.read_csv(file_path, sep = '\s+', names = column_names_short)
                df.insert(0, "subject", n)
                df.insert(1, "action_type", type)
                df.insert(2, "action", action)
                all_data_df = pd.concat([all_data_df, df], axis=0)
                count += 1
                print(f"[INFO]: Done reading file \t{folder_name}{action}.txt \t\t Read {round(count/total_iters*100, 2)}% complete")
        else:
            # type == "normal"
            for action in ["Bowing", "Clapping", "Handshaking", "Hugging", "Jumping", "Running", "Seating", "Standing", "Walking", "Waving"]:
                if n == 10 and action == "Waving":
                    # for some reason no data for sub10/normal/Waving
                    pass
                else:
                    file_path = "/Users/lukedavidson/Downloads/Vicon_Physical_Action_Data_Set/" + folder_name + action + ".txt"
                    df = pd.read_csv(file_path, sep = '\s+', names = column_names_short)
                    df.insert(0, "subject", n)
                    df.insert(1, "action_type", type)
                    df.insert(2, "action", action)
                    all_data_df = pd.concat([all_data_df, df], axis=0)
                    count += 1
                    print(f"[INFO]: Done reading file \t{folder_name}{action}.txt \t\t Read {round(count/total_iters*100, 2)}% complete")
print("[TRACE]: Done with reading process.")
print(f"[INFO]: Reading process took {round(time.time() - start_time, 2)} s")

# Combination of x, y, z coords function
def combo_xyz(df, limb):
    return (df[limb + "x"]**2 + df[limb + "y"]**2 + df[limb + "z"]**2)**0.5

# Insert combos of x, y, z coords
print("[TRACE]: Beginning xyz coordinate combination calculations and inserts...")
for limb, loc in limb_col_combo:
    all_data_df.insert(loc, limb + "c", combo_xyz(all_data_df, limb))
    print(f"[INFO]: Done creating combination for {limb}")
print("[TRACE]: Done with coordinate combination process.")
file_path = "/Users/lukedavidson/Documents/CS6140_FinalProj/all_data.csv"
print(f"[TRACE]: Saving all_data df to file path {file_path} ...")
all_data_df.to_csv(file_path)
print("[TRACE]: File successfully saved.")



# Normalize all_data_df to all_data_norm_df
all_data_norm_df = all_data_df.copy()
print("[TRACE]: Beginning normalization process...")
count = 0
start_time = time.time()
for sub in range(1, 11):
    print("\n*********************************")
    print(f"***** Normalizing subject {sub} *****")
    print("*********************************")
    for act in all_actions:
        subact = all_data_df.loc[(all_data_df["subject"] == sub) & (all_data_df["action"] == act)]
        for col in limb_col:
            # denom = ((subact[col]**2).sum())**0.5
            subact[col] += -subact[col].min()
            denom = subact[col].max()
            # print(denom)
            all_data_norm_df.loc[(all_data_norm_df["subject"] == sub) & (all_data_norm_df["action"] == act), col] = subact[col]/denom
        count += 1
        print(f"[INFO]: Done normalizing Subject: {sub}\t Action: {act} \t\tNorm {round(count/total_iters*100, 2)}% complete")
print(f"[TRACE]: Done with normalization process.")
print(f"[INFO]: Normalizing process took {round(time.time() - start_time, 2)} s")
file_path = "/Users/lukedavidson/Documents/CS6140_FinalProj/all_data_norm.csv"
print(f"[TRACE]: Saving all_data_norm df to file path {file_path} ...")
all_data_norm_df.to_csv(file_path)
print("[TRACE]: File successfully saved.")

