import torch
from ultralytics import YOLO


def load_yolo_model(weights_path, task):
    """
    Загружает модель YOLO из checkpoint.

    Args:
        weights_path: путь к файлу весов.
        task: тип задачи ('detect' или 'classify').

    Returns:
        Загруженная модель YOLO.
    """

    model = YOLO(str(weights_path), task=task)

    return model


def get_model_state(model):
    """
    Извлекает параметры нейронной сети.

    Возвращает state_dict модели на CPU,
    не изменяя исходную модель.
    """

    state_dict = model.model.state_dict()

    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in state_dict.items()
    }


def load_model_state(model, global_state):
    """
    Загружает глобальные параметры в локальную модель.

    strict=True гарантирует, что архитектуры моделей
    и наборы параметров полностью совпадают.
    """

    model.model.load_state_dict(
        global_state,
        strict=True
    )

    return model


def clone_model_state(state_dict):
    """
    Создаёт независимую копию набора параметров.
    """

    return {
        key: tensor.detach().cpu().clone()
        for key, tensor in state_dict.items()
    }


def compare_model_states(state_1, state_2):
    """
    Проверяет совместимость двух наборов параметров.
    """

    if state_1.keys() != state_2.keys():
        return False

    for key in state_1:
        if state_1[key].shape != state_2[key].shape:
            return False

        if state_1[key].dtype != state_2[key].dtype:
            return False

    return True