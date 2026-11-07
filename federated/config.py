from pathlib import Path


# Пути проекта

ROOT = Path(r"D:\fire_extinguisher_research")

CLIENTS = {
    "client_1": ROOT / "local" / "data" / "client_1",
    "client_2": ROOT / "local" / "data" / "client_2",
}

DATASET_PATH = ROOT / "dataset" / "dataset_final"

FEDERATED_PATH = ROOT / "federated"

RESULTS_PATH = ROOT / "results" / "federated"


# Параметры федеративного обучения

NUM_CLIENTS = 2

NUM_TIME_STEPS = 5

FEDAVG_ROUNDS = 5

LOCAL_EPOCHS = 2


# Параметры скользящего окна

WINDOW_SIZE = 2000

WINDOW_STEP = 720

NUM_ROUNDS = NUM_TIME_STEPS

# Параметры моделей

IMAGE_SIZE = 320

BATCH_SIZE = 4

DEVICE = 0

WORKERS = 0

# Воспроизводимость эксперимента

SEED = 42

DETERMINISTIC = True


# Классы моделей

DETECTION_CLASSES = {
    0: "fire_extinguisher"
}

CLASSIFICATION_CLASSES = {
    0: "absent",
    1: "present"
}