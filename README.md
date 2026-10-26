# SuperMimic Parkour

Learning highly dynamic motions through PPO exploration can take substantial training time. Inspired by SuperMimic and building on open-source projects, this implementation trained a policy to clear a 130 cm obstacle in just over 4 hours on a single GPU.

## Installation

```bash
cd supermimic_parkour
conda activate mjlab
python -m pip install -e .
```

## Motion Data

Data source: [PHP Parkour — motion_matching](https://github.com/amazon-far/php_parkour/tree/main/motion_matching).

Motion data is not included in this repository. Use `--dataset-dir` to specify the PHP bundle root directory. By default, training reads from `data/php_motion` relative to the current working directory. Organize the data as follows:

```text
php_motion/
  php-motions-low-step-v1/
    <name>_motion.npz
    <name>_terrain.npy
  php-motions-high-step-v1/
  php-motions-low-climb-76-v1/
  php-motions-high-climb-76-v1/
  php-motions-ultra-climb-130-v1/
```

## Train

```bash
python scripts/train_php_supermimic.py \
  --gpu-ids 0 \
  --log_dir logs/php_five_skills
```

## Play

```bash
python scripts/play_php_supermimic.py \
  --checkpoint logs/php_five_skills/last.pt \
  --device cuda:0 --port 8082
```

Open `http://127.0.0.1:8082` in your browser to view the policy alongside the reference motion.

### Demo

![SuperMimic Parkour policy playback](<doc/录屏 2026年10月26日 16时41分26秒.gif>)
