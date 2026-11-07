from pathlib import Path
import json
import torch
from ultralytics import YOLO

ROOT = Path(r"D:\fire_extinguisher_research")
CLIENT_PATH = ROOT / "local" / "data" / "client_2"
DATA_YAML = CLIENT_PATH / "data.yaml"
RESULTS = ROOT / "results"

EPOCHS = 30
IMAGE_SIZE = 320
BATCH_SIZE = 4
PATIENCE = 7
SEED = 42

RUN_NAME = "local_client_2_corrected"

print("CUDA available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available")

print("GPU:", torch.cuda.get_device_name(0))

if not DATA_YAML.exists():
    raise FileNotFoundError(DATA_YAML)


data_root = CLIENT_PATH.resolve()
train_dir = (data_root / "images" / "train").resolve()

expected_dir = (CLIENT_PATH / "images" / "train").resolve()

print("Корень датасета:", data_root)
print("Изображения для тренировки:", train_dir)

if train_dir != expected_dir:
    raise RuntimeError(
        f"Неверный путь: {train_dir}"
    )

if not train_dir.exists():
    raise FileNotFoundError(train_dir)

image_count = len(list(train_dir.iterdir()))
print("Training image count:", image_count)

if image_count != 4879:
    raise RuntimeError("Неверные значения")

output_dir = RESULTS / RUN_NAME

if output_dir.exists():
    raise FileExistsError(
        f"Путь уже существует: {output_dir}"
    )

print("\nНачнем тренировку второго клиента 2")

model = YOLO("yolo11n.pt")

model.train(
    data=str(DATA_YAML),
    epochs=EPOCHS,
    imgsz=IMAGE_SIZE,
    batch=BATCH_SIZE,
    patience=PATIENCE,
    device=0,
    workers=0,
    amp=False,
    seed=SEED,
    deterministic=True,
    project=str(RESULTS),
    name=RUN_NAME,
    exist_ok=False,
    save=True,
    plots=True,
    verbose=True
)

save_dir = Path(model.trainer.save_dir)
best_weights = save_dir / "weights" / "best.pt"
last_weights = save_dir / "weights" / "last.pt"

if not best_weights.exists():
    raise FileNotFoundError(best_weights)

trained_models = {
    "client_2_corrected": str(best_weights),
    "client_2_corrected_last": str(last_weights)
}

output_file = ROOT / "local" / "trained_models_corrected.json"

if output_file.exists():
    raise FileExistsError(output_file)

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(trained_models, f, indent=4, ensure_ascii=False)

print("\nОбучение завершено")
print("Лучшие веса:", best_weights)
print("Последние веса:", last_weights)
print("Путь сохранен в:", output_file)
