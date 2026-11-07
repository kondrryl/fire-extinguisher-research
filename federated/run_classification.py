from federated.federated_coordinator import run_federated_training


if __name__ == "__main__":

    print("=" * 65)
    print("Федеративное обучение: классификация")
    print("=" * 65)

    final_global_state = run_federated_training(
        task="classify"
    )

    print("\nОбучение классификатора завершено.")
    print(
        "Количество итоговых параметров:",
        len(final_global_state)
    )