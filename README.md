# Multi-review Fusion-in-Context

## Overview
This is the repository of the paper "Multi-Review Fusion-in-Context".

## FuseReviews Benchmark
For the FuseReviews benchmark, see [benchmark website](https://fusereviews.github.io/).

## Getting Started
To install all the required packages, run:
```
pip install -r requirements.txt
```

## Few-shot Experiments
To run the few shot experiments, run:
```
cd few_shot_experiments
python prompt_gpt.py --model-name <MODEL> --n-demos <N_DEMOS> --split <SPLIT> --openAI-key <API_KEY> --outdir /path/to/outdir
```

* `<MODEL>`- model name (e.g., gpt-4).
* `<N_DEMOS>` - number of few-shot examples.
* `<SPLIT>` - data split (any one of train, validation, or test).
* `<API_KEY>` - OpenAI key.


## Code for Finetuning Experiments Coming Soon!
To finetune Flan-T5 large, run:
```
cd finetuned_experiments
python -m src.run_experiments configs/train/train_flan_t5_large.json
```

To evaluate the finetuned model, update the `model_name_or_path` and `output_dir` in [evaluation configs](https://github.com/fusereviews/multi-review-fusion-in-context/tree/main/finetuned_experiments/configs/eval/eval_flan_t5_large.json) to point to the relative path of the finetuned checkpoint and directory where to store the results, accordingly. Then, run:
```
cd finetuned_experiments
python -m src.run_experiments configs/eval/eval_flan_t5_large.json
```


## Stay Updated
To receive the latest news and updates, please star or watch this repository.

## Contact
If you have any questions, suggestions, or would like to contribute in the meantime, please don't hesitate to contact us via the issues tab.

We appreciate your interest and patience. Stay tuned!
