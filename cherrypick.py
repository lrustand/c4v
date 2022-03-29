#!/usr/bin/env python
import pandas as pd
import numpy as np
from pathlib import Path
flops = {}
for val_test in ["val", "test"]:
    for acc_loss in ["acc", "loss"]:
        for small_normal in ["small", "normal"]:
            for ds in ["fish", "birds", "plants", "cifar", "aircrafts"]:
                df = []
                run = 1
                while(True):
                    path = f"runs/{val_test}/{acc_loss}/{run}/{small_normal}_{ds}.csv"
                    try:
                        new_df = pd.read_csv(path, index_col=0)
                        old_columns = new_df.columns
                        new_df = new_df.loc[:, new_df.var() > 0.00001]
                        new_columns = new_df.columns
                        df.append(new_df)
                        for column in old_columns:
                            if column not in new_columns:
                                full_column = f"{column},{val_test},{acc_loss},{small_normal}"
                                if full_column in flops:
                                    flops[full_column] += 1
                                else:
                                    flops[full_column] = 1
                    except:
                        break
                    run += 1
                if df:
                    df = pd.concat(df).groupby(level=0).mean()
                    path = f"runs/{val_test}/{acc_loss}/avgnzv/{small_normal}_{ds}.csv"
                    filepath = Path(path)
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    df.to_csv(path)
for flop in flops:
    print(f"{flop},{flops[flop]}")
