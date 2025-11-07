# 1 GPU
# accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=1 --main_process_port 29501 scripts/train_sd3.py --config config/grpo.py:general_ocr_sd3_1gpu
# 4 GPU
accelerate launch --config_file scripts/accelerate_configs/multi_gpu.yaml --num_processes=8 --main_process_port 29501 scripts/train_sd3_self.py --config config/grpo.py:general_ocr_sd3_8gpu
