# <img alt="icon" src="figures/icon.png" height=30> MUSE: Machine Unlearning Six-Way Evaluation for Language Models

### Conda Environment

To create a conda environment for Python 3.10, run:
```bash
conda env create -f environment.yml
conda activate py310
```

## 📘 Data & Target Models

Two corpora `News` and `Books` and the associated target models are available as follows:

| Domain | <div style="text-align: center">Target Model for Unlearning</div> | Dataset |
|----------|:------------------------------:|----------| 
| News | [Target model](https://huggingface.co/muse-bench/MUSE-News_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-News) |
| Books | [Target model](https://huggingface.co/muse-bench/MUSE-Books_target) | [Dataset](https://huggingface.co/datasets/muse-bench/MUSE-Books) | 

Before proceeding, load all the data from HuggingFace to the root of this repostiory by running the following instruction:
```
python load_data.py
```

## 🚀 Run unlearning baselines

To unlearn the target model using our baseline method, run `unlearn.py` in the `baselines` folder. Example scripts `baselines/scripts/unlearn_news.sh` and `scripts/unlearn_books.sh` in the `baselines` folder demonstrate the usage of `unlearn.py`. Here is an example:
```bash
algo="ga"
CORPUS="news"

python unlearn.py \
        --algo $algo \
        --model_dir $TARGET_DIR --tokenizer_dir 'meta-llama/Llama-2-7b-hf' \
        --data_file $FORGET --retain_data_file $RETAIN \
        --out_dir "./ckpt/$CORPUS/$algo" \
        --max_len $MAX_LEN --epochs $EPOCHS --lr $LR \
        --per_device_batch_size $PER_DEVICE_BATCH_SIZE
```

- `algo`: Unlearning algorithm to run (`ga`, `ga_gdr`, `ga_klr`, `npo`, `npo_gdr`, `npo_klr`, or `tv`).
- `model_dir`: Directory of the target model.
- `tokenizer_dir`: Directory of the tokenizer.
- `data_file`: Forget set.
- `retain_data_file`: Retain set for GDR/KLR regularizations if required by the algorithm.
- `out_dir`: Directory to save the unlearned model (default: `ckpt`).
- `max_len`: Maximum input length (default: 2048).
- `per_device_batch_size`, `epochs`, `lr`: Hyperparameters.

----
**Resulting models are saved in the `ckpt` folder as shown:**
```
ckpt
├── news/
│   ├── ga/
│   │   ├── checkpoint-102
│   │   ├── checkpoint-204
│   │   ├── checkpoint-306
│   │   └── ...
│   └── npo/
│       └── ...
└── books/
    ├── ga
    └── ...
```
# 🔍 Evaluation of Unlearned Models

To evaluate your unlearned model(s), run `eval.py` from the root of this repository with the following command-line arguments:

- `--model_dirs`: A list of directories containing the unlearned models. These can be either HuggingFace model directories or local storage paths.
- `--names`: A unique name assigned to each unlearned model in `--model_dirs`. The length of `--names` should match the length of `--model_dirs`.
- `--corpus`: The corpus to use for evaluation. Options are `news` or `books`.
- `--out_file`: The name of the output file. The file will be in CSV format, with each row corresponding to an unlearning method from `--model_dirs`, and columns representing the metrics specified by `--metrics`.
- `--tokenizer_dir` (Optional): The directory of the tokenizer. Defaults to `meta-llama/Llama-2-7b-hf`, which is the default tokenizer for LLaMA.
- `--metrics` (Optional): The metrics to evaluate. Options are `verbmem_f` (VerbMem Forget), `privleak` (PrivLeak), `knowmem_f` (KnowMem Forget), and `knowmem_r` (Knowmem Retain, i.e., Utility). Defaults to evaluating all these metrics.
- `--temp_dir` (Optional): The directory for saving intermediate computations. Defaults to `temp`.

### Example Command

Run the following command with placeholder values:

```bash
python eval.py \
  --model_dirs "jaechan-repo/model1" "jaechan-repo/model2" \
  --names "model1" "model2" \
  --corpus books \
  --out_file "out.csv"
```
