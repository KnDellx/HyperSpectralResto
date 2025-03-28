# Original DiffUn
python scripts/DiffUn_w.py --input_hsi data/A4_P1500.npz --save_dir "results/unmixing/" --model_config models/A4_P1500/model_config.yaml --in_channels 1 --range_t 0 --diffusion_steps 1000 --rescale_timesteps True --cache_H True
# setting --cache_H to be True for accelerating the computation
python scripts/DiffUn_w_re.py --input_hsi data/A4_P1500.npz --save_dir "results/unmixing/" --in_channels 1 --range_t 0 --diffusion_steps 1000 --rescale_timesteps True --cache_H True