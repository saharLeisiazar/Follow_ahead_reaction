
import numpy as np
import colorsys
import matplotlib.pyplot as plt

import matplotlib

fig, axs = plt.subplots(1)
colors = [colorsys.hsv_to_rgb(h, 1, 1) for h in np.linspace(0, 1, 13)]
cmap = matplotlib.colors.ListedColormap(colors)
pcm = axs.pcolormesh(np.random.random((1, 13)), cmap=cmap, vmin=0, vmax=13)

fig.colorbar(pcm, ax=axs, label='Time (s)')



plt.savefig('bar_plot.png')