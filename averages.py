#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path

for val_test in ["val", "test"]:
    for acc_loss in ["acc", "loss"]:
        for small_normal in ["small", "normal"]:
            for ds in ["fish", "birds", "plants", "cifar", "aircrafts"]:
                df = []
                run = 1
                while(True):
                    path = f"runs/{val_test}/{acc_loss}/{run}/{small_normal}_{ds}.csv"
                    try:
                        df.append(pd.read_csv(path, index_col=0))
                    except:
                        break
                    run += 1
                if df:
                    df = pd.concat(df).groupby(level=0).mean()
                    path = f"runs/{val_test}/{acc_loss}/avg/{small_normal}_{ds}.csv"
                    filepath = Path(path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path)
