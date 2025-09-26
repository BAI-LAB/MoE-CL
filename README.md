Here’s your document fully translated into English:

---

# MoE-CL: Mixture-of-Experts Continual Learning Framework

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

## Evaluation

Evaluate the trained model:

```bash
python evaluate_mtl5.py \
    --base_model ../model/Llama-2-7b-hf \
    --config configs/mtl5/moe-cl.json \
    --order order1 \
    --load_adapter_file agnews_finetuned
```

### Output Results

* Training results are saved in the `results/` directory
* Training logs are saved in the `logs/` directory

---

Would you like me to also make the **shell commands and file paths more generic** (e.g., not tied to `mtl5` or `Llama-2-7b-hf`) so the instructions can be reused for other models and configs?
