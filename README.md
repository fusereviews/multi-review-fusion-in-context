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
* `<SPLIT>` - data split (any one of train, validation, or test)
* `<API_KEY>` - OpenAI key


## Code for Finetuning Experiments Coming Soon!
We're excited to share that the code is on its way! 
Our team is working hard to ensure that we deliver high-quality and well-documented code for you to use. 

Here's what you can expect:
* **Comprehensive Documentation**: Clear instructions on how to set up and use the code.
* **Fully Tested**: Code that has been rigorously tested to ensure reliability and performance.

## Timeline
Our target for releasing the code is June 2024. Please stay tuned for updates!

## Stay Updated
To receive the latest news and updates, please star or watch this repository.

## Contact
If you have any questions, suggestions, or would like to contribute in the meantime, please don't hesitate to contact us via the issues tab.

We appreciate your interest and patience. Stay tuned!
