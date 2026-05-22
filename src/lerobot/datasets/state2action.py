import numpy as np
import pandas as pd
from pathlib import Path

root = Path("/home/allen/Allen/lerobot/dataset/data_5_9")
state_key = "observation.state"

for pq in sorted(root.glob("data/chunk-*/*.parquet")):
    df = pd.read_parquet(pq)
    df = df.sort_values(["episode_index", "frame_index"]).reset_index(drop=True)

    new_actions = np.empty((len(df), len(df[state_key].iloc[0])), dtype=np.float32)
    row = 0
    for _, sub in df.groupby("episode_index", sort=False):
        sub = sub.sort_values("frame_index")
        states = np.stack(sub[state_key].values)
        act = np.empty_like(states)
        act[:-1] = states[1:]
        act[-1] = states[-1]
        new_actions[row : row + len(sub)] = act
        row += len(sub)

    df["action"] = [new_actions[i] for i in range(len(df))]
    df.to_parquet(pq)