import pandas as pd


IN_PATH = "outputs/metrics/early_stop_simulation.csv"


def compute_detection_delay(df):
    results = []

    for ip, g in df.groupby("Src IP"):
        g = g.sort_values("window_start")

        attack_rows = g[g["target"] == 1]
        if len(attack_rows) == 0:
            continue

        attack_start = attack_rows.iloc[0]["window_start"]

        block_rows = g[g["action"] == "BLOCK"]
        if len(block_rows) == 0:
            continue

        block_time = block_rows.iloc[0]["window_start"]

        delay_sec = (
            pd.to_datetime(block_time) - pd.to_datetime(attack_start)
        ).total_seconds()

        results.append({
            "Src IP": ip,
            "attack_start": attack_start,
            "block_time": block_time,
            "delay_sec": delay_sec
        })

    return pd.DataFrame(results)


def main():
    df = pd.read_csv(IN_PATH)

    print("=== ACTION COUNTS ===")
    print(df["action"].value_counts())

    # Detection delay
    delay_df = compute_detection_delay(df)

    print("\n=== DETECTION DELAY ===")
    print(delay_df)

    if len(delay_df) > 0:
        print("\nMean delay:", delay_df["delay_sec"].mean())

    # False positive
    benign = df[df["target"] == 0]
    fp = benign[benign["action"].isin(["ALERT", "BLOCK"])]

    print("\n=== FALSE POSITIVE ===")
    print("FP count:", len(fp))
    print("Total benign:", len(benign))
    print("FP rate:", len(fp) / len(benign))


if __name__ == "__main__":
    main()
