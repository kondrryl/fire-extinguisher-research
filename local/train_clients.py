from pathlib import Path
from ultralytics import YOLO
import json
# Корень проекта
ROOT = Path(__file__).resolve().parents[1]

CLIENTS = {
    "client_1": ROOT / "local" / "data" / "client_1",
    "client_2": ROOT / "local" / "data" / "client_2",
}

RESULTS = ROOT / "results"

EPOCHS = 30
IMAGE_SIZE = 320
BATCH_SIZE = 4
PATIENCE = 7
SEED = 42
def train_client(client_name, client_path):
    print("\n" + "=" * 60)
    print(f"Обучение: {client_name}")
    print("=" * 60)
  
    model = YOLO("yolo11n.pt")

    results = model.train(
        data=str(client_path / "data.yaml"),
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
        name=f"local_{client_name}",
        exist_ok=False,
        save=True,
        plots=True,
        verbose=True
    )
    print(f"\nОбучение {client_name} завершено!")

    best_weights = (
        RESULTS
        / f"local_{client_name}"
        / "weights"
        / "best.pt"
    )
    print(f"Лучшая модель: {best_weights}")
    return best_weights
if __name__ == "__main__":
    trained_models = {}
    for client_name, client_path in CLIENTS.items():
        best_weights = train_client(
            client_name,
            client_path
        )
        trained_models[client_name] = str(best_weights)
        
    output_file = ROOT / "local" / "trained_models.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(
            trained_models,
            f,
            indent=4,
            ensure_ascii=False
        )
    print("\n" + "=" * 60)
    print("Локальные модели обучены!")
    print("=" * 60)
    for name, path in trained_models.items():
        print(f"{name}: {path}")