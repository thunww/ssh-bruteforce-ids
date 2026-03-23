from pathlib import Path
import pandas as pd

RAW_PATH = Path("data/raw/tuesday.csv")

def main():
    df = pd.read_csv(RAW_PATH, low_memory=False)

    raw_cols = [c.strip() for c in df.columns]

    print(f"Total columns read: {len(raw_cols)}")

    counts = {}
    for c in raw_cols:
        counts[c] = counts.get(c, 0) + 1

    dupes = {k: v for k, v in counts.items() if v > 1}

    print(f"Unique column names: {len(counts)}")
    print(f"Duplicate names count: {len(dupes)}")

    if dupes:
        print("\nDuplicate columns:")
        for k, v in sorted(dupes.items(), key=lambda x: (-x[1], x[0])):
            print(f"{k}: {v}")

if __name__ == "__main__":
    main()
