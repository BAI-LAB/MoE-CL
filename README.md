# MoE-CL

Mixture-of-Experts for Continual Learning on MTL5 benchmark.

## Environment Setup

```bash
pip install -r requirements.txt
```

## Model Preparation

Download the **Llama-2-7b-hf** model into the `../model/Llama-2-7b-hf/` directory.

## Training

Run the full continual learning training (including random initialization baseline + continual learning sequence **DBPedia → Amazon → Yahoo → AGNews**):

```bash
bash scripts/mtl5/run_moe-cl.sh
```

## Evaluation Metrics

After training, calculate the continual learning metrics (ACC, BWT, FWT):

```bash
# Calculate metrics for order1
python calculate_bwt_fwt.py \
    --log_file results/moe-cl/mtl5/order1/log.txt \
    --order order1

# With random initialization baseline for FWT calculation
python calculate_bwt_fwt.py \
    --log_file results/moe-cl/mtl5/order1/log.txt \
    --order order1 \
    --random_init_log results/moe-cl/mtl5/rand_init/log.txt
```

### Metrics Explanation

| Metric | Description |
|--------|-------------|
| **ACC** | Average accuracy across all tasks after learning the final task |
| **BWT** | Backward Transfer — measures forgetting (negative = forgetting occurred) |
| **FWT** | Forward Transfer — measures knowledge transfer to new tasks (positive = helpful) |

### Available Task Orders

- `order1`: DBPedia → Amazon → Yahoo → AGNews
- `order2`: DBPedia → Amazon → AGNews → Yahoo
- `order3`: Yahoo → Amazon → AGNews → DBPedia

## Output

- Training results and model checkpoints: `results/` directory
- Training logs: `logs/` directory
