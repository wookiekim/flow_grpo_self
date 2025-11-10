# 8 GPU
accelerate launch --config_file scripts/accelerate_configs/deepspeed_zero1.yaml --num_processes=8 --main_process_port 29501 scripts/train_flux_self.py --config config/grpo.py:general_ocr_flux_8gpu
