import torch


def fedavg(state_dicts, sample_counts):
    """
    Выполняет агрегацию параметров локальных моделей
    по алгоритму Federated Averaging (FedAvg).

    Args:
        state_dicts: список state_dict локальных моделей.
        sample_counts: количество обучающих наблюдений
                       у каждого клиента.

    Returns:
        Агрегированный глобальный state_dict.
    """

    if not state_dicts:
        raise ValueError("Список локальных моделей пуст.")

    if len(state_dicts) != len(sample_counts):
        raise ValueError(
            "Количество моделей не совпадает "
            "с количеством размеров выборок."
        )

    if any(count <= 0 for count in sample_counts):
        raise ValueError(
            "Размеры выборок должны быть положительными."
        )

    total_samples = sum(sample_counts)

    # Вычисляем вес каждого клиента.
    weights = [
        count / total_samples
        for count in sample_counts
    ]

    reference_keys = list(state_dicts[0].keys())

    # Проверяем совместимость локальных моделей.
    for client_state in state_dicts[1:]:
        if list(client_state.keys()) != reference_keys:
            raise ValueError(
                "Локальные модели имеют разные наборы параметров."
            )

    global_state = {}

    for key in reference_keys:

        tensors = [
            state[key]
            for state in state_dicts
        ]

        reference_tensor = tensors[0]

        # Проверяем размерности и типы параметров.
        for tensor in tensors[1:]:
            if tensor.shape != reference_tensor.shape:
                raise ValueError(
                    f"Несовпадение размерностей параметра: {key}"
                )

            if tensor.dtype != reference_tensor.dtype:
                raise ValueError(
                    f"Несовпадение типов параметра: {key}"
                )

        # Усредняем вещественные и комплексные параметры.
        if (
            torch.is_floating_point(reference_tensor)
            or torch.is_complex(reference_tensor)
        ):
            accumulation_dtype = (
                torch.complex128
                if torch.is_complex(reference_tensor)
                else torch.float64
            )

            averaged_tensor = torch.zeros(
                reference_tensor.shape,
                dtype=accumulation_dtype,
                device="cpu"
            )

            for tensor, weight in zip(tensors, weights):
                averaged_tensor += (
                    tensor.detach().cpu().to(accumulation_dtype)
                    * weight
                )

            global_state[key] = averaged_tensor.to(
                dtype=reference_tensor.dtype
            )

        else:
            # Целочисленные и логические буферы не усредняются арифметически.
            global_state[key] = (
                reference_tensor.detach().cpu().clone()
            )

    return global_state