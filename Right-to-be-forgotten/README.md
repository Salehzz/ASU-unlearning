# Right to be Forgotten

## Installation

We follow the [TOFU Benchmark](https://github.com/locuslab/tofu/?tab=readme-ov-file#installation) to install the required dependencies, please run the following commands:

```shell
conda create -n tofu python=3.10
conda activate tofu
conda install pytorch pytorch-cuda=11.8 -c pytorch -c nvidia
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit
pip install -r requirements.txt
pip install flash-attn --no-build-isolation
```


*All experiments are conducted on two NVIDIA A100 GPUs with 40GB of memory.*

## Fictitious unlearning scenario

**(1) ASU**

```shell
bash scripts/tofu/me_gd.sh
```

**(2) Baselines**

```shell
bash scripts/tofu/baselines.sh
```


## Continual unlearning scenario

**(1) ASU**

```shell
bash scripts/continual_tofu/asu.sh
```

**(2) Baselines**

```shell
bash scripts/continual_tofu/baselines.sh
```


## Real-world unlearning scenario

To evaluate LLMs on general tasks in real-world unlearning scenarios, we follow [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) to install the `lm-eval` package:
```shell
git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
```

**(1) ASU**

```shell
bash scripts/real_world/asu.sh
```

**(2) Baselines**
```shell
bash scripts/real_world/baselines.sh
```


## Acknowledgments

This repository is based on the codebase of the [A Closer Look at Machine Unlearning for Large Language Models](https://github.com/sail-sg/closer-look-LLM-unlearning). Thanks for their impressive works!

