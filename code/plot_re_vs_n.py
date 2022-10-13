import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import r2_score
dir = "./MODIFIED_SingleChainMC/outputs"
re_list = []
rg_list = []
x = []
old_re_val = 0;
for i in range(1,127):
	f = open(dir+"/"+"RE"+str(i)+".dat", "r")
	re_val = float(f.read())
	
	re_list.append(re_val)
	x.append(i)

	f = open(dir+"/"+"RG"+str(i)+".dat", "r")
	rg_list.append(float(f.read()))

fig, axs = plt.subplots(1, 2)

axs[0].set_xticklabels(x, fontsize=14)
axs[0].set_yticklabels(rg_list, fontsize=14)

axs[1].set_xticklabels(x, fontsize=14)
axs[1].set_yticklabels(re_list, fontsize=14)

axs[0].scatter(x, rg_list)
axs[0].set_xlabel('log N',  fontsize=16)
axs[0].set_ylabel(r'log $\langle Rg^2 \rangle $', fontsize=16)
axs[0].grid(True, which="both", ls="-", color='0.65')
axs[0].loglog()
axs[1].scatter(x[:len(re_list)], re_list)
axs[1].set_xlabel('log N',  fontsize=16)
axs[1].set_ylabel(r'log $\langle Re^2 \rangle $', fontsize=16)
axs[1].grid(True, which="both", ls="-", color='0.65')
axs[1].loglog()
plt.show()

plt.close()