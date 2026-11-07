from pathlib import Path
from sklearn.model_selection import train_test_split
import os
import shutil

ROOT = Path(__file__).resolve().parents[1]

DATASET = ROOT / "dataset" / "dataset_final"

OUTPUT = ROOT / "local" / "data"

IMAGES = DATASET / "images" / "train"
LABELS = DATASET / "labels" / "train"

SEED = 42


def is_positive(label_path):
    """Проверяем, есть ли в аннотации хотя бы один объект."""
    return bool(label_path.read_text(encoding="utf-8").strip())


def link_or_copy(source, destination):
    """Создаём жёсткую ссылку, а если нельзя — копируем файл."""
    destination.parent.mkdir(parents=True, exist_ok=True)

    if destination.exists():
        destination.unlink()

    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


image_files = sorted([
    p for p in IMAGES.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
])

if not image_files:
    raise RuntimeError("Обучающие изображения не найдены!")

labels = []
for image in image_files:
    label = LABELS / f"{image.stem}.txt"

    if not label.exists():
        raise FileNotFoundError(f"Не найдена аннотация: {label}")

    labels.append(is_positive(label))

client_1, client_2 = train_test_split(
    image_files,
    test_size=0.5,
    random_state=SEED,
    stratify=labels
)

clients = {
    "client_1": client_1,
    "client_2": client_2
}

for client_name, files in clients.items():

    client_root = OUTPUT / client_name

    for image in files:
        label = LABELS / f"{image.stem}.txt"

        image_dst = client_root / "images" / "train" / image.name
        label_dst = client_root / "labels" / "train" / label.name

        link_or_copy(image, image_dst)
        link_or_copy(label, label_dst)

    print(f"\n{client_name}:")
    print(f"Всего изображений: {len(files)}")
    print(f"Положительных: {sum(is_positive(LABELS / (p.stem + '.txt')) for p in files)}")
    print(f"Отрицательных: {sum(not is_positive(LABELS / (p.stem + '.txt')) for p in files)}")

    yaml_content = f"""
path: {client_root}
train: images/train
val: {DATASET / "images" / "val"}
test: {DATASET / "images" / "test"}

names:
  0: fire_extinguisher
"""

    (client_root / "data.yaml").write_text(
        yaml_content.strip() + "\n",
        encoding="utf-8"
    )

print("\nРазделение завершено!")
print(f"Общее количество изображений: {len(image_files)}")