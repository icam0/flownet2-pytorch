import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')

x = np.arange(0,20,1)
v = 5.

wx1 = 1./20.
wx2 = 1./10.

wt1 = v*wx1
wt2 = v*wx2

y1_t1 = np.sin(x*wx1*2*np.pi)
y2_t1 = np.sin(x*wx2*2*np.pi)

y1_t2 = np.sin(x*wx1 *2*np.pi + 1*wt1*2*np.pi )
y2_t2 = np.sin(x*wx2 *2*np.pi + 1*wt2*2*np.pi )

current_palette = sns.color_palette()

blue = current_palette[0]
orange = current_palette[1]
green  = current_palette[2]
gray = current_palette[-3]

fig,axes = plt.subplots(3,2)

axes[0,0].plot(x,y1_t1,'.-',color=blue,label=r'$y_{1}$')
axes[1,0].plot(y2_t1,'.-',color=orange,label=r'$y_{2}$')
axes[2,0].plot(y1_t1+y2_t1,'.-',color=green,label=r'$y_{1} + y_{2}$')

axes[0,1].plot(y1_t2,'.-',color=blue)
axes[1,1].plot(y2_t2,'.-',color=orange)
axes[2,1].plot(y1_t2+y2_t2,'.-',color=green)

axes[0,1].scatter(x[0],y1_t2[0],color='r',zorder=20)
axes[0,0].scatter(x[5],y1_t1[5],color='r',zorder=20)

axes[1,1].scatter(x[0],y2_t2[0],color='r',zorder=20)
axes[1,0].scatter(x[5],y2_t1[5],color='r',zorder=20)

axes[2,1].scatter(x[0],y1_t2[0]+y2_t2[0],color='r',zorder=20)
axes[2,0].scatter(x[5],y1_t1[5]+y2_t1[5],color='r',zorder=20)

axes[0,0].set_title(r'$t_{1}$')
axes[0,1].set_title(r'$t_{2}$')

axes[0,0].set_ylabel(r'$y_{1}$')
axes[1,0].set_ylabel(r'$y_{2}$')
axes[2,0].set_ylabel(r'$y_{1} + y_{2}$')

for ax in axes.flatten():
    ax.set_ylim([-2.,2.])
    #ax.axhline(0,color='k',zorder=-10)
sns.despine(offset=0,bottom=False)
#fig.legend()

fig.savefig('./freq_vel.pdf',dpi=600)
plt.show()