import matplotlib.pyplot as plt
import pandas as pd
import sys

files = sys.argv[1:]

for f in files:
    df = pd.read_csv(f, index_col=0)
    df.plot(kind="line")
    plt.savefig(f[:-4] + "_csv.png")
    plt.close("all")
