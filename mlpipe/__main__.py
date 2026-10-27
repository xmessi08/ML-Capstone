import sys
from mlpipe.train import train_and_evaluate
from mlpipe.predict import main as predict_main

if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "train":
        train_and_evaluate()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        # Сдвигаем аргументы для argparse в predict.py
        sys.argv.pop(1)
        predict_main()
    else:
        print("Use 'train' or 'predict'")
