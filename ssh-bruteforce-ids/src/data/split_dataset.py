import pandas as pd


def classwise_time_split(
    df: pd.DataFrame,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
):
    df = df.sort_values("window_start").reset_index(drop=True)

    attack_df = df[df["target"] == 1].sort_values("window_start").reset_index(drop=True)
    benign_df = df[df["target"] == 0].sort_values("window_start").reset_index(drop=True)

    def _split_one(sub_df: pd.DataFrame):
        n = len(sub_df)
        train_end = int(n * train_ratio)
        val_end = int(n * (train_ratio + val_ratio))

        train_part = sub_df.iloc[:train_end]
        val_part = sub_df.iloc[train_end:val_end]
        test_part = sub_df.iloc[val_end:]
        return train_part, val_part, test_part

    atk_train, atk_val, atk_test = _split_one(attack_df)
    ben_train, ben_val, ben_test = _split_one(benign_df)

    train_df = pd.concat([atk_train, ben_train], axis=0).sort_values("window_start").reset_index(drop=True)
    val_df = pd.concat([atk_val, ben_val], axis=0).sort_values("window_start").reset_index(drop=True)
    test_df = pd.concat([atk_test, ben_test], axis=0).sort_values("window_start").reset_index(drop=True)

    return train_df, val_df, test_df