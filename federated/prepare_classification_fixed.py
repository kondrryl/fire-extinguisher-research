import os
import random
import math

from config import (
    CLIENTS,
    DATASET_PATH,
    RESULTS_PATH,
    WINDOW_SIZE,
    WINDOW_STEP,
    NUM_ROUNDS,
    SEED,
)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Отдельная директория — старые данные не затрагиваем
OUTPUT_PATH = (
    RESULTS_PATH
    / "classification"
    / "data_fixed"
)


def get_label_class(label_path):
    """
    0 — объект отсутствует.
    1 — объект присутствует.
    """

    content = label_path.read_text(
        encoding="utf-8"
    ).strip()

    if not content:
        return 0

    for line in content.splitlines():

        parts = line.split()

        if len(parts) != 5:
            raise ValueError(
                f"Некорректная разметка: {label_path}"
            )

        class_id = int(parts[0])

        if class_id != 0:
            raise ValueError(
                f"Неизвестный класс: {label_path}"
            )

        coordinates = [
            float(value) for value in parts[1:]
        ]

        if not all(
            math.isfinite(value)
            for value in coordinates
        ):
            raise ValueError(
                f"Некорректные координаты: {label_path}"
            )

    return 1


def get_records(images_dir, labels_dir):

    images = sorted(
        path for path in images_dir.iterdir()
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )

    records = []

    for image_path in images:

        label_path = (
            labels_dir / f"{image_path.stem}.txt"
        )

        if not label_path.is_file():
            raise FileNotFoundError(
                f"Не найдена разметка: {label_path}"
            )

        class_id = get_label_class(label_path)

        records.append((image_path, class_id))

    return records


def create_class_directories(base_path):

    for class_name in ("absent", "present"):
        (base_path / class_name).mkdir(
            parents=True,
            exist_ok=True
        )


def link_records(records, output_path):

    counts = {
        0: 0,
        1: 0
    }

    for image_path, class_id in records:

        class_name = (
            "absent" if class_id == 0 else "present"
        )

        target_path = (
            output_path
            / class_name
            / image_path.name
        )

        os.link(image_path, target_path)

        counts[class_id] += 1

    return counts


def prepare_client(client_name, client_path, val_records):

    images_dir = client_path / "images" / "train"
    labels_dir = client_path / "labels" / "train"

    records = get_records(
        images_dir,
        labels_dir
    )

    if len(records) < WINDOW_SIZE:
        raise ValueError(
            f"Недостаточно данных у {client_name}"
        )

    # Сохраняем тот же порядок, что использовался при первоначальной подготовке M1 и M2.
    rng = random.Random(SEED)
    rng.shuffle(records)

    starts = [
        min(
            i * WINDOW_STEP,
            len(records) - WINDOW_SIZE
        )
        for i in range(NUM_ROUNDS)
    ]

    print(f"\n{client_name}")
    print(f"Всего изображений: {len(records)}")

    for round_number, start in enumerate(starts, start=1):

        window = records[
            start:start + WINDOW_SIZE
        ]

        round_dir = (
            OUTPUT_PATH
            / client_name
            / f"round_{round_number}"
        )

        if round_dir.exists():
            raise FileExistsError(
                f"Каталог уже существует: {round_dir}"
            )

        train_dir = round_dir / "train"
        val_dir = round_dir / "val"

        create_class_directories(train_dir)
        create_class_directories(val_dir)

        train_counts = link_records(
            window,
            train_dir
        )

        val_counts = link_records(
            val_records,
            val_dir
        )

        print(
            f"Раунд {round_number}: "
            f"train={sum(train_counts.values())} "
            f"(absent={train_counts[0]}, "
            f"present={train_counts[1]}) | "
            f"val={sum(val_counts.values())} "
            f"(absent={val_counts[0]}, "
            f"present={val_counts[1]})"
        )


def main():

    val_images = (
        DATASET_PATH / "images" / "val"
    )

    val_labels = (
        DATASET_PATH / "labels" / "val"
    )

    val_records = get_records(
        val_images,
        val_labels
    )

    if len(val_records) != 121:
        raise ValueError(
            f"Ожидалось 121 val-изображение, "
            f"получено {len(val_records)}"
        )

    print("Общая validation-выборка:")
    print(f"Изображений: {len(val_records)}")
    print(
        "Отрицательных:",
        sum(class_id == 0 for _, class_id in val_records)
    )
    print(
        "Положительных:",
        sum(class_id == 1 for _, class_id in val_records)
    )

    for client_name, client_path in CLIENTS.items():

        prepare_client(
            client_name,
            client_path,
            val_records
        )

    print(
        "\nПодготовка исправленных "
        "classification-данных завершена."
    )


if __name__ == "__main__":
    main()