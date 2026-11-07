import json
import torch

from pathlib import Path
from ultralytics import YOLO

from federated.config import (
    ROOT,
    RESULTS_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    WORKERS,
)

from federated.local_trainer import create_local_model


# 1. Пути к данным и моделям
dataset_yaml = (
    ROOT / "local" / "data" / "client_1" / "data.yaml"
)

models = {
    "centralized": (
        ROOT / "results" / "centralized_baseline-2"
        / "weights" / "best.pt"
    ),

    "local_client_1": (
        ROOT / "results" / "local_client_1-3"
        / "weights" / "best.pt"
    ),

    "local_client_2": (
        ROOT / "results" / "local_client_2_corrected"
        / "weights" / "best.pt"
    ),
}


# 2. Загрузка глобальной федеративной модели
checkpoint_path = (
    RESULTS_PATH
    / "detect"
    / "global_models"
    / "time_5"
    / "round_5"
    / "global_state.pt"
)

checkpoint = torch.load(
    checkpoint_path,
    map_location="cpu",
    weights_only=False
)

federated_model = create_local_model(
    task="detect",
    global_state=checkpoint["state_dict"]
)


# 3. Формирование списка моделей
models_to_evaluate = {
    name: YOLO(str(path))
    for name, path in models.items()
}

models_to_evaluate["federated"] = federated_model


# 4. Оценка на тестовой выборке
results = {}

for name, model in models_to_evaluate.items():

    print("\n" + "=" * 65)
    print(f"Тестирование: {name}")
    print("=" * 65)

    metrics = model.val(
        data=str(dataset_yaml),
        split="test",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        conf=0.001,
        iou=0.7,
        max_det=300,
        device=DEVICE,
        workers=WORKERS,
        half=False,
        plots=True,
        verbose=False,
        project=str(ROOT / "results"),
        name=f"final_test_{name}",
        exist_ok=False,
    )

    results[name] = {
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "map50": float(metrics.box.map50),
        "map50_95": float(metrics.box.map),
    }

    print(f"Precision:  {metrics.box.mp:.4f}")
    print(f"Recall:     {metrics.box.mr:.4f}")
    print(f"mAP50:      {metrics.box.map50:.4f}")
    print(f"mAP50-95:   {metrics.box.map:.4f}")


# 5. Сохранение итоговых результатов
output_path = ROOT / "results" / "final_test_comparison.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n" + "=" * 65)
print("Итоговое сравнение")
print("=" * 65)

for name, values in results.items():
    print(f"\n{name}")

    for metric, value in values.items():
        print(f"{metric}: {value:.4f}")

print(f"\nРезультаты сохранены: {output_path}")