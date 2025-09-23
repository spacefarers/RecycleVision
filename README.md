# RecycleVision

RecycleVision is a lightweight computer vision training scaffold focused on classifying waste into recyclable categories using a MobileNetV3-Small backbone.

## Project Goals
- Provide a compact baseline model suitable for resource-constrained deployments.
- Streamline experimentation on public garbage and recycling datasets.
- Offer reproducible training loops and configuration-driven workflows.

## Repository Layout
- `main.py` — CLI entry point that wires configs and kicks off training.
- `src/recyclevision/config.py` — Dataclasses describing experiment configuration.
- `src/recyclevision/data.py` — Data loading and transform utilities built around `ImageFolder`.
- `src/recyclevision/model.py` — MobileNetV3-Small model factory with a slim classification head.
- `src/recyclevision/train.py` — Training + evaluation loops with checkpointing.
- `configs/default.yaml` — Example configuration for quick-start runs.
- `requirements.txt` — Core dependencies.

## Getting Started
1. Create a virtual environment and install dependencies:
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Arrange your dataset in `ImageFolder` layout:
   ```
   data/
     garbage-classification/
       train/
         cardboard/
         glass/
         metal/
         paper/
         plastic/
         trash/
       val/  # optional, otherwise 20% of train becomes validation
   ```
3. Launch training with the provided config:
   ```bash
   python main.py --config configs/default.yaml
   ```
   or supply arguments directly:
   ```bash
   python main.py --data-root ./data/garbage-classification --num-classes 6
   ```

Checkpoints are written to `./artifacts/checkpoints` by default. Use `torch.load` to resume or export the model.

Evaluation runs every `eval_interval_epochs` (default 5). Set `training.eval_interval_epochs = 1` if you prefer validation after each epoch. Mixup (default `training.mixup_alpha = 0.4`) and richer image augmentation are enabled to curb overfitting; disable by setting the alpha to `0`.

## Gradio Inference UI
- Launch an interactive demo once you have a trained checkpoint:
  ```bash
  python gradio_app.py --data-root ./data/garbage-classification --num-classes 6 \
      --checkpoint checkpoints/best-epoch25.pt
  ```
- Add `--share` for a public Gradio link, or use `--server-name` / `--server-port` to bind locally.
- Use the flag button to capture hard examples; a normalized `image_size`×`image_size` copy of the model input is written to `flags/processed/` (override with `--flag-dir`).

## MobileNetV3 Notes
- Uses torchvision's `mobilenet_v3_small` weights initialised from ImageNet for faster convergence.
- Final classifier is trimmed to 256 hidden units with dropout to keep the parameter count low (<2.6M params).
- Mixed precision can be toggled to reduce memory footprint on capable GPUs.

## Public Waste Classification Datasets
Below are popular datasets suitable for training or fine-tuning. Always review licenses and preprocessing requirements.

| Dataset | Size / Classes | Link | Notes |
| --- | --- | --- | --- |
| TrashNet | ~2527 images, 6 classes | [https://github.com/garythung/trashnet](https://github.com/garythung/trashnet) | Classic benchmark with cardboard/glass/metal/paper/plastic/trash. Small; good for prototyping.
| TACO (Trash Annotations in Context) | 1,500+ images, 60 classes | [http://tacodataset.org](http://tacodataset.org) | Instance segmentation labels; convert to classification by cropping or using bounding boxes.
| Waste Classification Data | 15,240 images, 12 classes | [https://www.kaggle.com/datasets/sapal6/waste-classification-data](https://www.kaggle.com/datasets/sapal6/waste-classification-data) | Larger Kaggle dataset with organic vs recyclable types.
| Garbage Classification | 29,140 images, 6 classes | [https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification](https://www.kaggle.com/datasets/asdasdasasdas/garbage-classification) | Well-organized train/test split; aligns with default config.
| ZeroWaste Dataset | 100k images, 20+ classes | [https://zerowaste.ai/dataset](https://zerowaste.ai/dataset) | Retail waste dataset; includes product packaging. Consider class merging for MobileNet baseline.

Complement the above with synthetic augmentation, dataset balancing, and domain-specific captures to improve generalization.

## Next Steps
- Integrate experiment tracking (TensorBoard/W&B) for metric logging.
- Add quantization-aware training for deployment on microcontrollers.
- Extend `data.py` with dataset downloading scripts or wrappers per source.
