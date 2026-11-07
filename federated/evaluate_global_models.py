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


def load_global_state(task):
    """
    Загружает итоговые глобальные параметры
    после последнего раунда FedAvg.
    """

    checkpoint_path = (
        RESULTS_PATH
        / task
        / "global_models"
        / "time_5"
        / "round_5"
        / "global_state.pt"
    )

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Глобальная модель не найдена: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False
    )

    global_state = checkpoint["state_dict"]

    print(f"\nЗагружена глобальная модель: {task}")
    print("Путь:", checkpoint_path)
    print("Количество тензоров:", len(global_state))

    return global_state


def evaluate_classification():
    """
    Оценивает глобальный классификатор
    на общей validation-выборке.
    """

    print("\n" + "=" * 65)
    print("Оценка федеративного классификатора")
    print("=" * 65)

    global_state = load_global_state("classify")

    model = create_local_model(
        task="classify",
        global_state=global_state
    )

    dataset_path = (
        RESULTS_PATH
        / "classification"
        / "data_fixed"
        / "client_1"
        / "round_1"
    )

    if not dataset_path.exists():
        raise FileNotFoundError(dataset_path)

    metrics = model.val(
        data=str(dataset_path),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        verbose=True,
        project=str(RESULTS_PATH),
        name="federated_classification",
        exist_ok=False
    )

    results = {
        key: float(value)
        for key, value in metrics.results_dict.items()
    }

    print("\nРезультаты классификации:")

    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    return results


def evaluate_detection():
    """
    Оценивает глобальный детектор
    на общей validation-выборке.
    """

    print("\n" + "=" * 65)
    print("Оценка федеративного детектора")
    print("=" * 65)

    global_state = load_global_state("detect")

    model = create_local_model(
        task="detect",
        global_state=global_state
    )

    dataset_yaml = (
        RESULTS_PATH
        / "detection"
        / "data"
        / "client_1"
        / "round_1"
        / "data.yaml"
    )

    if not dataset_yaml.exists():
        raise FileNotFoundError(dataset_yaml)

    metrics = model.val(
        data=str(dataset_yaml),
        split="val",
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        device=DEVICE,
        workers=WORKERS,
        plots=True,
        verbose=True,
        project=str(RESULTS_PATH),
        name="federated_detection",
        exist_ok=False
    )

    results = {
        key: float(value)
        for key, value in metrics.results_dict.items()
    }

    print("\nРезультаты обнаружения:")

    for key, value in results.items():
        print(f"{key}: {value:.6f}")

    return results


def main():

    print("\n" + "=" * 65)
    print("Оценка федеративных глобальных моделей")
    print("=" * 65)

    classification_results = evaluate_classification()
    detection_results = evaluate_detection()

    final_results = {
        "classification": classification_results,
        "detection": detection_results
    }

    output_path = (
        RESULTS_PATH
        / "federated_metrics.json"
    )

    with open(output_path, "w", encoding="utf-8") as file:
        json.dump(
            final_results,
            file,
            ensure_ascii=False,
            indent=4
        )

    print("\n" + "=" * 65)
    print("Оценка завершена")
    print("=" * 65)

    print("\nМетрики сохранены:")
    print(output_path)


if __name__ == "__main__":
    main()