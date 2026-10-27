import argparse
import pandas as pd
from mlpipe.data import load

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    
    df = load(args.data)
    # Простейший прогноз (например, нули или единицы)
    preds = [0] * len(df)
    
    out_df = pd.DataFrame({"prediction": preds})
    out_df.to_csv(args.out, index=False)
    print(f"Predictions saved to {args.out}")

if __name__ == "__main__":
    main()
