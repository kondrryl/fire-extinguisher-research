import torch 
import numpy as np
from federated.config import ( RESULTS_PATH, IMAGE_SIZE, BATCH_SIZE, DEVICE, WORKERS, )
from federated.local_trainer import create_local_model


checkpoint_path = ( RESULTS_PATH / "detect" / "global_models" / "time_5" / "round_5" / "global_state.pt" )
checkpoint = torch.load( checkpoint_path, map_location="cpu", weights_only=False )
global_state = checkpoint["state_dict"]
model = create_local_model( task="detect", global_state=global_state )

dataset_yaml = ( RESULTS_PATH / "detection" / "data" / "client_1" / "round_1" / "data.yaml" )

metrics = model.val( data=str(dataset_yaml), split="val", imgsz=IMAGE_SIZE, batch=BATCH_SIZE, device=DEVICE, workers=WORKERS, plots=True, verbose=False, project=str(RESULTS_PATH), name="federated_detection_standard_check", exist_ok=False, )

print("\nОсновные метрики") 
print("=" * 65)
print(f"Precision:  {metrics.box.mp:.4f}") 
print(f"Recall:     {metrics.box.mr:.4f}") 
print(f"mAP50:      {metrics.box.map50:.4f}") 
print(f"mAP50-95:   {metrics.box.map:.4f}")

curves = metrics.curves_results
confidence_f1 = curves[1][0] 
f1_values = curves[1][1][0]
confidence_p = curves[2][0] 
precision_values = curves[2][1][0]
confidence_r = curves[3][0] 
recall_values = curves[3][1][0]
best_idx = np.argmax(f1_values) 
best_conf = confidence_f1[best_idx]
p_idx = np.argmin(np.abs(confidence_p - best_conf)) 
r_idx = np.argmin(np.abs(confidence_r - best_conf))
print("\nАнализ кривых") 
print("=" * 65)
print(f"Оптимальный confidence: {best_conf:.4f}") 
print(f"Максимальный F1:        {f1_values[best_idx]:.4f}") 
print(f"Precision:              {precision_values[p_idx]:.4f}") 
print(f"Recall:                 {recall_values[r_idx]:.4f}")

#6. Путь к сохранённым графикам
print("\nГрафики сохранены в:") 
print(RESULTS_PATH / "federated_detection_standard_check")