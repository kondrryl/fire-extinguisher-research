import json
import torch

from federated.config import (
    RESULTS_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    WORKERS,
)

from federated.local_trainer import create_local_model



CLASSIFICATION_DATA = (
    RESULTS_PATH
    / "classification"
    / "data_fixed"
    / "client_1"
    / "round_1"
)

DETECTION_DATA = (
    RESULTS_PATH
    / "detection"
    / "data"
    / "client_1"
    / "round_1"
    / "data.yaml"
)



results = {
    "classification": {},
    "detection": {},
}

# 3. Оценка пяти временных шагов
for time_step in range(1, 6):

    print("\n" + "=" * 60)
    print(f"Временной шаг {time_step}")
    print("=" * 60)

    for task in ["classify", "detect"]:

        print(f"\nМодель: {task}")

        # Путь к глобальным параметрам
        checkpoint_path = (
            RESULTS_PATH
            / task
            / "global_models"
            / f"time_{time_step}"
            / "round_5"
            / "global_state.pt"
        )

        # Загружаем глобальное состояние
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False
        )

        global_state = checkpoint["state_dict"]

        # Создаём соответствующую архитектуру
        model = create_local_model(
            task=task,
            global_state=global_state
        )

        # Выбираем общую валидационную выборку
        if task == "classify":
            dataset = CLASSIFICATION_DATA
        else:
            dataset = DETECTION_DATA

        # Оценка без дополнительного обучения
        metrics = model.val(
            data=str(dataset),
            split="val",
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            device=DEVICE,
            workers=WORKERS,
            plots=True,
            verbose=False,
            project=str(RESULTS_PATH),
            name=f"temporal_{task}_time_{time_step}",
            exist_ok=False,
        )

        # Сохраняем метрики
        if task == "classify":

            values = {
                "top1": float(metrics.top1),
                "top5": float(metrics.top5),
            }

            results["classification"][f"time_{time_step}"] = values

            print(f"Top-1 accuracy: {metrics.top1:.4f}")
            print(f"Top-5 accuracy: {metrics.top5:.4f}")

        else:

            values = {
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
            }

            results["detection"][f"time_{time_step}"] = values

            print(f"Precision:  {metrics.box.mp:.4f}")
            print(f"Recall:     {metrics.box.mr:.4f}")
            print(f"mAP50:      {metrics.box.map50:.4f}")
            print(f"mAP50-95:   {metrics.box.map:.4f}")

# 4. Сохранение результатов
output_path = RESULTS_PATH / "federated_temporal_metrics.json"

with open(output_path, "w", encoding="utf-8") as file:
    json.dump(
        results,
        file,
        ensure_ascii=False,
        indent=4
    )

print("\n" + "=" * 60)
print("Оценка завершена")
print("=" * 60)

print(f"Результаты сохранены: {output_path}")