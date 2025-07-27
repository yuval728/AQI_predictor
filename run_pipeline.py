from src.pipeline import AQITrainingPipeline, quick_train
from src.config import DATASET_PATH, TRAIN_CSV, VAL_CSV, TEST_CSV
import pandas as pd
import warnings
warnings.filterwarnings("ignore")


if __name__ == "__main__":
    # Run the basic usage example
    results = quick_train(
        dataset_path=DATASET_PATH,
        train_csv=pd.read_csv(TRAIN_CSV),
        val_csv=pd.read_csv(VAL_CSV),
        test_csv=pd.read_csv(TEST_CSV)
    )
    # print("Results:", results)