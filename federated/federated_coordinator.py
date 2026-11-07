from pathlib import Path

import torch
from ultralytics import YOLO
from ultralytics.nn.tasks import ClassificationModel, DetectionModel

from federated.config import (
    CLIENTS,
    RESULTS_PATH,
    NUM_TIME_STEPS,
    FEDAVG_ROUNDS,
    IMAGE_SIZE,
)

from federated.fedavg import fedavg
from federated.local_trainer import train_local_model


def initialize_global_model(task):
    """
    Создаёт начальную глобальную модель.

    Используются предобученные веса YOLO.
    Для каждой задачи создаётся отдельная архитектура.
    """

    if task == "classify":

        pretrained = YOLO("yolo11n-cls.pt")

        model = ClassificationModel(
            cfg="yolo11n-cls.yaml",
            ch=3,
            nc=2,
            verbose=False
        )

    elif task == "detect":

        pretrained = YOLO("yolo11n.pt")

        model = DetectionModel(
            cfg="yolo11n.yaml",
            ch=3,
            nc=1,
            verbose=False
        )

    else:
        raise ValueError(f"Неизвестная задача: {task}")

    # Переносим совместимые предобученные веса.
    model.load(pretrained.model, verbose=True)

    # Извлекаем начальные глобальные параметры.
    global_state = {
        key: tensor.detach().cpu().clone()
        for key, tensor in model.state_dict().items()
    }

    print("\nНачальная глобальная модель создана.")
    print("Задача:", task)
    print("Количество тензоров:", len(global_state))

    return global_state


def save_global_state(
    task,
    time_step,
    fedavg_round,
    global_state,
    total_samples
):
    """
    Сохраняет глобальные параметры после FedAvg.
    """

    output_dir = (
        RESULTS_PATH
        / task
        / "global_models"
        / f"time_{time_step}"
        / f"round_{fedavg_round}"
    )

    if output_dir.exists():
        raise FileExistsError(
            f"Результаты уже существуют: {output_dir}"
        )

    output_dir.mkdir(
        parents=True,
        exist_ok=False
    )

    output_file = output_dir / "global_state.pt"

    torch.save(
        {
            "task": task,
            "time_step": time_step,
            "fedavg_round": fedavg_round,
            "total_samples": total_samples,
            "state_dict": global_state,
        },
        output_file
    )

    print("Глобальные параметры сохранены:")
    print(output_file)


def run_federated_training(task):
    """
    Запускает полный федеративный цикл.

    Для каждого временного шага выполняется
    заданное количество раундов FedAvg.
    """

    if task not in ("classify", "detect"):
        raise ValueError(
            "Задача должна быть classify или detect."
        )

    print("\n" + "=" * 65)
    print("Запуск федеративного обучения")
    print("Задача:", task)
    print("=" * 65)

    # 1. Инициализация глобальной модели
    global_state = initialize_global_model(task)

    clients = list(CLIENTS.keys())
    # 2. Последовательность временных шагов
    for time_step in range(1, NUM_TIME_STEPS + 1):

        print("\n" + "#" * 65)
        print(f"Временной шаг {time_step}/{NUM_TIME_STEPS}")
        print("#" * 65)
        # 3. Раунды FedAvg

        for fedavg_round in range(1, FEDAVG_ROUNDS + 1):

            print("\n" + "-" * 60)
            print(
                f"Раунд FEDAVG "
                f"{fedavg_round}/{FEDAVG_ROUNDS}"
            )
            print("-" * 60)

            local_results = []

            # 4. Обучение клиентов
            for client_id in clients:

                result = train_local_model(
                    task=task,
                    client_id=client_id,
                    time_step=time_step,
                    fedavg_round=fedavg_round,
                    global_state=global_state
                )

                local_results.append(result)

            # 5. Сбор локальных параметров
            local_states = [
                result["state_dict"]
                for result in local_results
            ]

            sample_counts = [
                result["sample_count"]
                for result in local_results
            ]

            # 6. Проверка количества наблюдений
            print("\nКоличество наблюдений клиентов:")

            for result in local_results:
                print(
                    result["client_id"],
                    result["sample_count"]
                )

            # 7. Агрегация FedAvg
            print("\nВыполняется агрегация FedAvg...")

            global_state = fedavg(
                local_states,
                sample_counts
            )

            print("Глобальные параметры обновлены.")

            # 8. Сохранение глобальной модели
            save_global_state(
                task=task,
                time_step=time_step,
                fedavg_round=fedavg_round,
                global_state=global_state,
                total_samples=sum(sample_counts)
            )

        print(
            f"\nВременной шаг {time_step} завершён."
        )

    print("\n" + "=" * 65)
    print("Федеративное обучение завершено")
    print("Задача:", task)
    print("=" * 65)

    return global_state