import shutil
import os

counter = 0
# count_bad = 0

CURRENT_DATASET_PATH = "./Check_Internal_En/"
NEW_DATASET_PATH = "./full_dataset_2/"

for i in range(1, 533):
    path_curr = CURRENT_DATASET_PATH + str(i)
    for filename in os.listdir(path_curr):
        if not filename[0] == "F":
            f = open(path_curr + "/" + filename, "r")
            vals = f.read().split("\n")
            # if not vals[-1].split(',')[0] == 'Density' and not vals[-2].split(',')[0] == 'Density':
            # 	count_bad+=1
            # 	continue

            shutil.copy(path_curr + "/" + filename, NEW_DATASET_PATH)
            os.rename(NEW_DATASET_PATH + filename, str(counter) + ".txt")
            counter += 1
print(counter)
# print(count_bad)
