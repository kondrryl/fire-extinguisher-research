import os
import random
import math

from config import (
    ROOT,
    CLIENTS,
    DATASET_PATH,
    RESULTS_PATH,
    WINDOW_SIZE,
    WINDOW_STEP,
    NUM_ROUNDS,
    SEED,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

OUTPUT_PATH = RESULTS_PATH / "detection" / "data"


def validate_label(label_path):
    """
    Проверяет YOLO-разметку.

    Пустой файл означает отсутствие объектов.
    Непустой содержит координаты bounding boxes.
    """

    content = label_path.read_text(encoding="utf-8").strip()

    if not content:
        return 0

    instance_count = 0

    for line in content.splitlines():

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Некорректное число значений: {label_path}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise ValueError(
                f"Неизвестный класс в {label_path}: {class_id}"
            )

        coordinates = [float(value) for value in parts[1:]]

        if not all(math.isfinite(value) for value in coordinates):
            raise ValueError(
                f"Некорректные координаты: {label_path}"
            )

        x, y, width, height = coordinates

        if not (
            0 <= x <= 1
            and 0 <= y <= 1
            and 0 < width <= 1
            and 0 < height <= 1
        ):
            raise ValueError(
                f"Координаты вне допустимого диапазона: {label_path}"
            )

        instance_count += 1

    return instance_count


def prepare_client(client_name, client_path):

    images_dir = client_path / "images" / "train"
    labels_dir = client_path / "labels" / "train"

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    if not images:
        raise RuntimeError(
            f"Изображения не найдены: {images_dir}"
        )

    records = []
    total_instances = 0
    negative_images = 0

    for image_path in images:

        label_path = labels_dir / f"{image_path.stem}.txt"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Не найдена разметка: {label_path}"
            )

        instances = validate_label(label_path)

        records.append((image_path, label_path))

        total_instances += instances

        if instances == 0:
            negative_images += 1

    if len(records) < WINDOW_SIZE:
        raise ValueError(
            f"Недостаточно изображений у {client_name}"
        )

    # Детерминированное перемешивание
    rng = random.Random(SEED)
    rng.shuffle(records)

    # Формирование окон
    starts = [
        min(
            i * WINDOW_STEP,
            len(records) - WINDOW_SIZE
        )
        for i in range(NUM_ROUNDS)
    ]

    if len(set(starts)) != NUM_ROUNDS:
        raise ValueError("Получились повторяющиеся окна")

    print(f"\n{client_name}")
    print(f"Всего изображений: {len(records)}")
    print(f"Всего объектов: {total_instances}")
    print(f"Изображений без объектов: {negative_images}")

    for round_number, start in enumerate(starts, start=1):

        window = records[start:start + WINDOW_SIZE]

        round_dir = (
            OUTPUT_PATH
            / client_name
            / f"round_{round_number}"
        )

        if round_dir.exists():
            raise FileExistsError(
                f"Каталог уже существует: {round_dir}. "
                "Перезапись запрещена."
            )

        images_output = round_dir / "images" / "train"
        labels_output = round_dir / "labels" / "train"

        images_output.mkdir(parents=True)
        labels_output.mkdir(parents=True)

        round_instances = 0
        round_negatives = 0

        for image_path, label_path in window:

            image_target = images_output / image_path.name
            label_target = labels_output / label_path.name

            os.link(image_path, image_target)
            os.link(label_path, label_target)

            instances = validate_label(label_path)

            round_instances += instances

            if instances == 0:
                round_negatives += 1

        # Создание YAML-конфигурации окна
        yaml_content = f"""path: {round_dir.as_posix()}

train: images/train
val: {(DATASET_PATH / "images" / "val").as_posix()}
test: {(DATASET_PATH / "images" / "test").as_posix()}

nc: 1
names:
  0: fire_extinguisher
"""

        data_yaml = round_dir / "data.yaml"
        data_yaml.write_text(yaml_content, encoding="utf-8")

        print(
            f"Раунд {round_number}: "
            f"{len(window)} изображений | "
            f"объектов: {round_instances} | "
            f"без объектов: {round_negatives}"
        )


def main():

    for client_name, client_path in CLIENTS.items():
        prepare_client(client_name, client_path)

    print("\nПодготовка detection-данных завершена.")


if __name__ == "__main__":
    main()