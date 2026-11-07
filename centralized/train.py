from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

DATASET = ROOT / "dataset" / "dataset_final"

RESULTS = ROOT / "results"


if not DATASET.exists():
    raise FileNotFoundError(
        f"Датасет не найден: {DATASET}"
    )


data_yaml = DATASET / "data_local.yaml"

data_yaml.write_text(
    f"""
path: {DATASET.resolve()}
train: images/train
val: images/val
test: images/test

names:
  0: fire_extinguisher
""".strip() + "\n"
)

print("Датасет:", DATASET)
print("Конфигурация:", data_yaml)


print("\nЗагрузка YOLO11n...")

model = YOLO("yolo11n.pt")


print("\nНачинаем централизованное обучение")

results = model.train(

    data=str(data_yaml),

    epochs=30,
    imgsz=320,
    batch=4,

    patience=7,

    device="cpu",
    workers=0,
    amp=False,

    seed=42,
    deterministic=True,

    project=str(RESULTS),
    name="centralized_baseline",
    exist_ok=False,

    save=True,
    plots=True,
    verbose=True
)


best_weights = (
    RESULTS /
    "centralized_baseline" /
    "weights" /
    "best.pt"
)

print("\nОбучение завершено!")
print("Лучшие веса:", best_weights)