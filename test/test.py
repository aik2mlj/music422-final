"""
Music 422

Plotting tool for BS.1116 Test Results

 MUSIC 422 listening tests (HW6)
-----------------------------------------------------------------------
 © 2009-25 Marina Bosi -- All rights reserved
-----------------------------------------------------------------------
"""

import matplotlib.pyplot as plt
from sdgplot import plotSDG


dirs = [
    "castanets/",
    "glockenspiel/",
    "harpsichord/",
    "spgm/",
    "pannedCello/",
]
fig128, ax128 = plotSDG(dirs, "128kbps_ours")
plt.show()

fig128, ax128 = plotSDG(dirs, "128kbps_bl")
plt.show()

fig96, ax96 = plotSDG(dirs, "96kbps_ours")
plt.show()

fig96, ax96 = plotSDG(dirs, "96kbps_bl")
plt.show()
