from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel, DetectionModel 

from federated.config import (
    ROOT,
    RESULTS_PATH,
    IMAGE_SIZE,
    BATCH_SIZE,
    DEVICE,
    WORKERS,
    SEED,
    DETERMINISTIC,
    LOCAL_EPOCHS,
    WINDOW_SIZE,
)

from federated.model_utils import compare_model_states


def get_dataset_path(task, client_id, time_step):
    """
    Возвращает путь к локальному окну клиента.
    """

    if task == "classify":
        return (
            RESULTS_PATH
            / "classification"
            / "data_fixed"
            / client_id
            / f"round_{time_step}"
        )

    if task == "detect":
        return (
            RESULTS_PATH
            / "detection"
            / "data"
            / client_id
            / f"round_{time_step}"
        )

    raise ValueError(f"Неизвестная задача: {task}")


def get_training_data_path(task, dataset_path):
    """
    Возвращает путь к данным в формате Ultralytics.
    """

    if task == "classify":
        return dataset_path

    if task == "detect":
        return dataset_path / "data.yaml"

    raise ValueError(f"Неизвестная задача: {task}")


def count_training_images(task, dataset_path):
    """
    Подсчитывает количество обучающих изображений.

    Для FedAvg учитывается количество изображений,
    а не количество объектов на изображениях.
    """

    if task == "classify":

        train_path = dataset_path / "train"

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        count = sum(
            1
            for path in train_path.rglob("*")
            if path.is_file()
            and path.suffix.lower() in image_extensions
        )

    elif task == "detect":

        train_path = dataset_path / "images" / "train"

        image_extensions = {".jpg", ".jpeg", ".png", ".bmp"}

        count = sum(
            1
            for path in train_path.iterdir()
            if path.is_file()
            and path.suffix.lower() in image_extensions
        )

    else:
        raise ValueError(f"Неизвестная задача: {task}")

    return count


def create_local_model(task, global_state):
    """
    Создаёт локальную модель той же архитектуры,
    что и глобальная модель координатора,
    и загружает глобальные параметры.
    """

    if task == "classify":

        yolo = YOLO("yolo11n-cls.pt")

        model = ClassificationModel(
            cfg="yolo11n-cls.yaml",
            ch=3,
            nc=2,
            verbose=False
        )

        # Повторяем инициализацию координатора.
        model.load(yolo.model, verbose=True)

        model.load_state_dict(
            global_state,
            strict=True
        )

    elif task == "detect":

        yolo = YOLO("yolo11n.pt")

        model = DetectionModel(
            cfg="yolo11n.yaml",
            ch=3,
            nc=1,
            verbose=False
        )

        # Повторяем инициализацию координатора.
        model.load(yolo.model, verbose=True)

        model.load_state_dict(
            global_state,
            strict=True
        )

    else:
        raise ValueError(f"Неизвестная задача: {task}")

    yolo.model = model

    return yolo

def train_local_model(
    task,
    client_id,
    time_step,
    fedavg_round,
    global_state
):
    """
    Выполняет локальное обучение одного клиента.

    Получает глобальные параметры,
    обучает модель на локальном окне
    и возвращает обновлённые параметры координатору.
    """

    print("\n" + "=" * 60)
    print(f"Задача: {task}")
    print(f"Клиент: {client_id}")
    print(f"Временной шаг: {time_step}")
    print(f"Раунд FedAvg: {fedavg_round}")
    print("=" * 60)
    # 1. Определяем локальный датасет

    dataset_path = get_dataset_path(
        task,
        client_id,
        time_step
    )

    if not dataset_path.exists():
        raise FileNotFoundError(
            f"Датасет не найден: {dataset_path}"
        )

    data_path = get_training_data_path(
        task,
        dataset_path
    )

    if not data_path.exists():
        raise FileNotFoundError(
            f"Файл или каталог данных не найден: {data_path}"
        )
    # 2. Подсчитываем локальные наблюдения

    sample_count = count_training_images(
        task,
        dataset_path
    )

    if sample_count != WINDOW_SIZE:
        raise ValueError(
            f"Ожидалось {WINDOW_SIZE} изображений, "
            f"найдено {sample_count}."
        )

    print(f"Локальных изображений: {sample_count}")
    # 3. Создаём локальную модель

    yolo = create_local_model(
        task,
        global_state
    )
    # 4. Формируем каталог результатов

    project_path = (
        RESULTS_PATH
        / task
        / "training"
        / f"time_{time_step}"
        / f"round_{fedavg_round}"
    )

    run_name = client_id

    output_path = project_path / run_name

    # Защита от перезаписи результатов.
    if output_path.exists():
        raise FileExistsError(
            f"Каталог результатов уже существует: {output_path}"
        )

    # 5. Локальное обучение

    print("\nНачинается локальное обучение...")

    yolo.train(
        data=str(data_path),

        epochs=LOCAL_EPOCHS,
        batch=BATCH_SIZE,
        imgsz=IMAGE_SIZE,

        device=DEVICE,
        workers=WORKERS,

        optimizer="AdamW",
        lr0=0.002,

        pretrained=True,

        patience=LOCAL_EPOCHS+1,

        amp=False,
        seed=SEED,
        deterministic=DETERMINISTIC,

        val=False,
        save=True,
        plots=False,
        verbose=False,

        project=str(project_path),
        name=run_name,
        exist_ok=False,
    )

    # 6. Получаем последний checkpoint

    last_checkpoint = Path(yolo.trainer.last)

    if not last_checkpoint.exists():
        raise FileNotFoundError(
            f"Последний checkpoint не найден: {last_checkpoint}"
        )

    print("\nПоследний checkpoint:")
    print(last_checkpoint)

    # 7. Загружаем last.pt

    trained_model = YOLO(
        str(last_checkpoint),
        task=task
    )

    local_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in trained_model.model.state_dict().items()
    }

    # 8. Проверяем совместимость параметров

    if not compare_model_states(
        global_state,
        local_state
    ):
        raise ValueError(
            "Полученные параметры несовместимы "
            "с глобальной моделью."
        )

    print("\nЛокальное обучение завершено.")
    print(f"Получено тензоров: {len(local_state)}")
    print(f"Размер локальной выборки: {sample_count}")

    # 9. Возвращаем параметры координатору

    return {
        "client_id": client_id,
        "state_dict": local_state,
        "sample_count": sample_count,
        "checkpoint": str(last_checkpoint),
    }