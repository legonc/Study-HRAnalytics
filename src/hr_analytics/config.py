# src/hr_analytics/config.py

RANDOM_STATE = 42
TEST_SIZE = 0.25

# пути к данным
DATA_RAW_PATH = "data/raw"
DATA_PROCESSED_PATH = "data/processed"

# имена файлов
TRAIN_JOB_FILE = "train_job_satisfaction_rate.csv"
TRAIN_QUIT_FILE = "train_quit.csv"
TEST_FEATURES_FILE = "test_features.csv"
TEST_JOB_TARGET_FILE = "test_target_job_satisfaction_rate.csv"
TEST_QUIT_TARGET_FILE = "test_target_quit.csv"