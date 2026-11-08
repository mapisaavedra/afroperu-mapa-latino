import matplotlib.pyplot as plt
from config import INK, MUTED

def setup():
    plt.rcParams.update({
        "figure.dpi": 110, "savefig.dpi": 160, "savefig.bbox": "tight",
        "font.size": 10.5, "axes.titlesize": 12.5, "axes.titleweight": "bold",
        "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
        "xtick.color": INK, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": "#e6e5df",
        "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    })
