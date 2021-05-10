
import os
os.chdir('/home/yuxi/PycharmProjects/pythonProject/gyoto_tutorial')
from plot_ray import *
from plot_bh import *
import numpy as np
os.chdir('/home/yuxi/Workshop/gyoto_shadow')

spins = [-1,0,0.5,1.0]
incs = [0,30,60,90,120]
# spins, incs = [0,1], [0,90]
# res=40
res=32

fig, axs = plt.subplots(len(incs), len(spins), figsize=(10,12))

for i_row, inc in enumerate(incs):
    for i_col, spin in enumerate(spins):
        Pt(spin,inc,1).shadow_image(res=res, ax=axs[i_row,i_col])
        axs[i_row,i_col].set_title('a=%.1f i=%.0f' % (spin, inc))

fig.tight_layout()
plt.savefig('test.png')
plt.show()

Pt(1,45,1).grid(np.arange(-0.1, 0.02, 0.01))

def t(n):
    i_start = np.arange(10., 180., 30.)
    i_finish = np.array(i_start)+30.
    for a in np.arange(0., 1.1, 0.1):
        for inc in np.arange(i_start[n], i_finish[n], 5.):
            bhs=Pt(a,inc,1.)
            bhs.shadow_image(0.15, 500)
