"""
Fine-tuning the library models for sequence to sequence.
"""

import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
import re
import datasets
import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

# from accelerate import Accelerator

import transformers
from filelock import FileLock
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    MBart50Tokenizer,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version, is_offline_mode
from transformers.utils.versions import require_version
from src.compute_metrics import compute_rouge_metrics, compute_meteor_metrics, compute_highlights_metrics
from src.freeze_embeds import freeze_embeds
from src.predictions_analyzer import PredictionsAnalyzer

from src.preprocessor import Preprocessor, get_special_tokens_constants

import evaluate
import torch
# torch.autograd.set_detect_anomaly(True) 
# torch.backends.cuda.matmul.allow_tf32 = True



# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.19.0.dev0")

require_version("datasets>=1.8.0",
                "To fix: pip install -r examples/pytorch/summarization/requirements.txt")

logger = logging.getLogger(__name__)

try:
    nltk.data.find("tokenizers/punkt")
except (LookupError, OSError):
    if is_offline_mode():
        raise LookupError(
            "Offline mode: run this script without TRANSFORMERS_OFFLINE first to download nltk data files"
        )
    with FileLock(".lock") as lock:
        nltk.download("punkt", quiet=True)

# A list of all multilingual tokenizer which require lang attribute.
MULTILINGUAL_TOKENIZERS = [
    MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast]


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={
            "help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={
            "help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

    load_in_8bit: bool = field(
        default=False,
        metadata={
            "help": "whether to load model in 8bit precision."
        },
    )


    device_map: str = field(
        default=None,
        metadata={
            "help": "When loading in 8bit precision, maps which layers are saved on which GPUs/CPU. If not passed and load_in_8bit is True, will be set to \"auto\"."
        },
    )

    resize_position_embeddings: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to automatically resize the position embeddings if `max_source_length` exceeds "
            "the model's position embeddings."
        },
    )
    
    freeze_embeds: bool = field(
        default=False
    )
    
    min_length: int = field(
        default=None
    )
    
    length_penalty: float = field(
        default=None
    )
    
    early_stopping: bool = field(
        default=False
    )
    
    no_repeat_ngram_size: int = field(
        default=None
    )

    local_radius: int = field(
        default=None
    )
    
    global_block_size: int = field(
        default=None
    )
    
    encoder_attention_type: int = field(
        default=None
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    experiment_type: str = field(default=None)
    lang: str = field(default=None, metadata={
                      "help": "Language id for summarization."})
    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    text_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the full texts (for summarization)."},
    )
    summary_column: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the column in the datasets containing the summaries (for summarization)."},
    )
    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a jsonlines or csv file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (rouge) on "
            "(a jsonlines or csv file)."
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (rouge) on " "(a jsonlines or csv file)."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default="", metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )

    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the decoder_start_token_id."
            "Useful for multilingual models like mBART where the first generated token"
            "needs to be the target language token (Usually it is the target language token)"
        },
    )
    
    add_global_attention: bool = field(
        default=False
    )
    
    add_global_attention_on_highlights: bool = field(
        default=False
    )
    
    add_global_attention_on_highlighted_words: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to add global attention not only on the highlight_start and highlight_end tokens, but also on the highlighted tokens themselves"
        }
    )
    
    should_preprocess_add_highlights: bool = field(
        default=True,
        metadata={
            "help": "Decides whether to add highlight tokens or not"
        }
    )

    
    should_add_reviews_delim: bool = field(
        default=True,
        metadata={
            "help": "Decides whether to add delimeter between reviews in the input"
        }
    )

    
    should_preprocess_only_sents_with_highlights: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to keep only sentences with highlights"
        }
    )
    
    should_preprocess_keep_only_highlights: bool = field(
        default=False,
        metadata={
            "help": "Decides whether to keep only highlights"
        }
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError(
                "Need either a dataset name or a training/validation file.")
        else:
            if self.train_file is not None:
                extension = self.train_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`train_file` should be a csv or a json file."
            if self.validation_file is not None:
                extension = self.validation_file.split(".")[-1]
                assert extension in [
                    "csv", "json"], "`validation_file` should be a csv or a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


summarization_name_mapping = {
    "amazon_reviews_multi": ("review_body", "review_title"),
    "big_patent": ("description", "abstract"),
    "cnn_dailymail": ("article", "highlights"),
    "orange_sum": ("text", "summary"),
    "pn_summary": ("article", "summary"),
    "psc": ("extract_text", "summary_text"),
    "samsum": ("dialogue", "summary"),
    "thaisum": ("body", "summary"),
    "xglue": ("news_body", "news_title"),
    "xsum": ("document", "summary"),
    "wiki_summary": ("article", "highlights"),
}


def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
    if (len(sys.argv) == 3 or len(sys.argv) == 2) and sys.argv[-1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file, (when passing 2 and the second is the json file - this is distributed training)
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[-1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if model_args.load_in_8bit and model_args.device_map is None:
        logger.info(f"passed load_in_8bit=True but didn't pass device_map. Will set device_map to 'auto'.")
        model_args.device_map = 'auto'

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            extension = data_args.test_file.split(".")[-1]
        raw_datasets = load_dataset(
            extension,
            data_files=data_files,
            cache_dir=model_args.cache_dir,
            use_auth_token=True if model_args.use_auth_token else None,
        )
    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # NEW from original script (enables sending config from arguments, necessary for LED)
    model_args_dict = {}
    model_args_dict.update(model_args.__dict__)
    model_args_dict['max_length'] = data_args.max_target_length  # We must add max_length when setting min_length
    model_args_dict = { k: v for k,v in model_args_dict.items() if v is not None}  # Important otherwise it might override default values

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        **model_args_dict # NEW from original script (enables sending config)
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None
    )

    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        device_map=model_args.device_map,
        load_in_8bit=model_args.load_in_8bit
    )  

    # NEW from original script
    is_t5_model = model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ] or model.config.model_type == 't5'  # Necessary when loading model from directory
    if data_args.source_prefix is None and is_t5_model:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is the expected, e.g. with "
            "`--source_prefix 'summarize: ' `"
        )

    # NEW from original script
    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # NEW from original script
    special_tokens_constants = get_special_tokens_constants(is_t5_model)
    preprocessor = Preprocessor(prefix=prefix, 
                                special_tokens_constants=special_tokens_constants, 
                                should_add_highlights=data_args.should_preprocess_add_highlights, 
                                should_add_reviews_delim=data_args.should_add_reviews_delim, 
                                only_sents_with_highlights=data_args.should_preprocess_only_sents_with_highlights, 
                                keep_only_highlights=data_args.should_preprocess_keep_only_highlights)

    # for the only-highlights or no-highlights case, it is important that we can still get the full input with highlights to properly calculate the faithfulness score.
    regular_highlights_preprocessor = Preprocessor(prefix=prefix, 
                                                   special_tokens_constants=special_tokens_constants, 
                                                   should_add_highlights=True, 
                                                   should_add_reviews_delim=data_args.should_add_reviews_delim, 
                                                   only_sents_with_highlights=False, 
                                                   keep_only_highlights=False)



    # NEW from original script
    if not all(special_tkn in tokenizer.additional_special_tokens for special_tkn in list(special_tokens_constants.values())):
        tokenizer.add_special_tokens({'additional_special_tokens': list(special_tokens_constants.values())})

    model.resize_token_embeddings(len(tokenizer))

    # NEW from original script
    if model_args.freeze_embeds:
        freeze_embeds(model)

    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(
                data_args.lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError(
            "Make sure that `config.decoder_start_token_id` is correctly defined")

    if (
        hasattr(model.config, "max_position_embeddings")
        and model.config.max_position_embeddings < data_args.max_source_length
    ):
        if model_args.resize_position_embeddings is None:
            logger.warning(
                f"Increasing the model's number of position embedding vectors from {model.config.max_position_embeddings} "
                f"to {data_args.max_source_length}."
            )
            model.resize_position_embeddings(data_args.max_source_length)
        elif model_args.resize_position_embeddings:
            model.resize_position_embeddings(data_args.max_source_length)
        else:
            raise ValueError(
                f"`--max_source_length` is set to {data_args.max_source_length}, but the model only has {model.config.max_position_embeddings}"
                f" position encodings. Consider either reducing `--max_source_length` to {model.config.max_position_embeddings} or to automatically "
                "resize the model's position encodings by passing `--resize_position_embeddings`."
            )

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info(
            "There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    if isinstance(tokenizer, tuple(MULTILINGUAL_TOKENIZERS)):
        assert (
            data_args.lang is not None
        ), f"{tokenizer.__class__.__name__} is a multilingual tokenizer which requires --lang argument"

        tokenizer.src_lang = data_args.lang
        tokenizer.tgt_lang = data_args.lang

        # For multilingual translation models like mBART-50 and M2M100 we need to force the target language token
        # as the first generated token. We ask the user to explicitly provide this as --forced_bos_token argument.
        forced_bos_token_id = (
            tokenizer.lang_code_to_id[data_args.forced_bos_token] if data_args.forced_bos_token is not None else None
        )
        model.config.forced_bos_token_id = forced_bos_token_id

    # Get the column names for input/target.
    dataset_columns = summarization_name_mapping.get(
        data_args.dataset_name, None)
    if data_args.text_column is None:
        text_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        text_column = data_args.text_column
        if text_column not in column_names:
            raise ValueError(
                f"--text_column' value '{data_args.text_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.summary_column is None:
        summary_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        summary_column = data_args.summary_column
        if summary_column not in column_names:
            raise ValueError(
                f"--summary_column' value '{data_args.summary_column}' needs to be one of: {', '.join(column_names)}"
            )

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    # NEW from original script
    def preprocess_function(examples):
        # Orig summarization dataset
        if data_args.dataset_name is not None:
            inputs, targets = [], []
            
            for i in range(len(examples[text_column])):
                if examples[text_column][i] is not None and examples[summary_column][i] is not None:
                    inputs.append(examples[text_column][i])
                    targets.append(examples[summary_column][i])

            inputs = [prefix + inp for inp in inputs]            
        else:
            inputs, targets = [], []
            regular_highlights_inputs = [] # in the cases when it's only highlights or no highlights, we still need to full reviews+highlights input for the faithfulness calculations
            for i in range(len(examples[text_column])):
                # NEW from original script
                curr_input = preprocessor.preprocess_input({key:value[i] for key,value in examples.items()})
                inputs.append(curr_input)
                curr_output = preprocessor.preprocess_output(examples['summary_text'][i], curr_input)
                targets.append(curr_output)
                curr_regular_inputs = regular_highlights_preprocessor.preprocess_input({key:value[i] for key,value in examples.items()})
                regular_highlights_inputs.append(curr_regular_inputs)

        model_inputs = tokenizer(
            inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                targets, max_length=max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]

        global_attention_mask = []
        if data_args.add_global_attention:
            for input_ids in tqdm(model_inputs['input_ids']):
                curr_global_attention_mask = [0 for _ in range(len(input_ids))]
                curr_global_attention_mask[0] = 1

                tkns_with_global_attention = [preprocessor.special_tokens_constants[tkn_key] for tkn_key in ['highlight_start', 'highlight_end']]
                ids_with_global_attention = [special_id for special_id in tokenizer.additional_special_tokens_ids if tokenizer.convert_ids_to_tokens(special_id) in tkns_with_global_attention]
                # ids_with_global_attention = tokenizer.additional_special_tokens_ids

                highlight_start_tkn_id = tokenizer.convert_tokens_to_ids(preprocessor.special_tokens_constants['highlight_start'])
                highlight_end_tkn_id = tokenizer.convert_tokens_to_ids(preprocessor.special_tokens_constants['highlight_end'])


                if data_args.add_global_attention_on_highlights:
                    highlight_began_flag = False
                    for input_id_idx, input_id in enumerate(input_ids):
                        # Put attention on highlight tokens
                        if input_id in ids_with_global_attention: #AVIVSL: play with this (regarding the other special tokens)
                            curr_global_attention_mask[input_id_idx] = 1
                        if data_args.add_global_attention_on_highlighted_words:
                            if input_id == highlight_start_tkn_id:
                                highlight_began_flag = True
                            elif input_id == highlight_end_tkn_id:
                                highlight_began_flag = False
                            elif highlight_began_flag:
                                curr_global_attention_mask[input_id_idx] = 1

                global_attention_mask.append(curr_global_attention_mask)
            model_inputs['global_attention_mask'] = global_attention_mask

        model_inputs["clustered_alignments"] = examples['clustered_alignments']
        model_inputs["full_inputs"] = examples['full_inputs']
        model_inputs["regular_highlights_inputs"] = regular_highlights_inputs
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            max_train_samples = min(
                len(train_dataset), data_args.max_train_samples)
            train_dataset = train_dataset.select(range(max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )



    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = raw_datasets["validation"]
        if data_args.max_eval_samples is not None:
            max_eval_samples = min(
                len(eval_dataset), data_args.max_eval_samples)
            eval_dataset = eval_dataset.select(range(max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            max_predict_samples = min(
                len(predict_dataset), data_args.max_predict_samples)
            predict_dataset = predict_dataset.select(
                range(max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            # NEW from original_script
            prep_predict_dataset = predict_dataset.map( 
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = - \
        100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=label_pad_token_id,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )

    # Metric
    rouge_metric = evaluate.load("rouge") 
    meteor_metric = evaluate.load('meteor')


    # faithfulness_metric = fic_evaluation.HighlightsFaithfulnessEvaluator()
    # coverage_metric = fic_evaluation.HighlightsCoverageEvaluator()
    # if torch.cuda.is_available():
    #     coverage_metric.to(device="cuda:0")

    def postprocess_text(inputs, preds, labels, preprocessor, tokenizer):
        all_special_tkns = sum([special_tkns if type(special_tkns)==list else [special_tkns] for special_tkns in tokenizer.special_tokens_map.values()], [])
        all_special_tkns_input = [special_tkn for special_tkn in all_special_tkns if not special_tkn in [preprocessor.special_tokens_constants["highlight_start"], preprocessor.special_tokens_constants["highlight_end"], preprocessor.special_tokens_constants["review_delim"]]]
        inputs = [re.sub(r'|'.join(map(re.escape, all_special_tkns_input)), '', input) for input in inputs] # remove the special tokens (but keep highlight_start, highlight_end, and review_delim)
        preds = [re.sub(r'|'.join(map(re.escape, all_special_tkns)), '', pred) for pred in preds] # remove the special tokens
        labels = [re.sub(r'|'.join(map(re.escape, all_special_tkns)), '', label) for label in labels] # remove the special tokens

        # split into the separate reviews
        inputs = [input.split(preprocessor.special_tokens_constants["review_delim"]) for input in inputs]


        inputs = [[rev.strip() for rev in input] for input in inputs]
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return inputs, preds, labels

    def compute_metrics(eval_preds, is_regular: bool, clustered_alignments, full_inputs, regular_highlights_inputs, preprocessor=preprocessor):
        inputs, labels, preds = eval_preds.inputs, eval_preds.label_ids, eval_preds.predictions
        if isinstance(preds, tuple):
            preds = preds[0]

        # Replace -100 in the inputs as we can't decode them.
        inputs = np.where(inputs != -100, inputs, tokenizer.pad_token_id)

        decoded_inputs = tokenizer.batch_decode(inputs)
        decoded_preds = tokenizer.batch_decode(preds)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels)

        # if the case when the input consists solely of the highlights or doesn't have any of the highlights, the use thr regular highlights inputs
        if not is_regular:
            decoded_inputs = regular_highlights_inputs

        # Some simple post-processing
        decoded_inputs, decoded_preds, decoded_labels = postprocess_text(
            decoded_inputs, decoded_preds, decoded_labels, preprocessor, tokenizer)


        result = compute_rouge_metrics(decoded_preds, decoded_labels, rouge_metric, prefix="full_gold")
        result.update(compute_rouge_metrics(decoded_preds, decoded_labels, rouge_metric, prefix="full_gold_content_", should_filter_function_words=True))
        result.update(compute_meteor_metrics(decoded_preds, decoded_labels, meteor_metric, prefix="full_gold_"))

        prediction_lens = [np.count_nonzero(
            pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)

        return result

    # NEW from original script (if training compute only necessary metrics, otherwise compute more)
    is_regular = data_args.should_preprocess_add_highlights and not data_args.should_preprocess_only_sents_with_highlights and not data_args.should_preprocess_keep_only_highlights
    if training_args.do_predict:
        curr_compute_metrics = lambda *args, **kwargs: compute_metrics(*args, **kwargs, is_regular=is_regular, clustered_alignments=prep_predict_dataset["clustered_alignments"], full_inputs=prep_predict_dataset["full_inputs"], regular_highlights_inputs=prep_predict_dataset["regular_highlights_inputs"], preprocessor=regular_highlights_preprocessor)    
    else:
        curr_compute_metrics = lambda *args, **kwargs: compute_metrics(*args, **kwargs, is_regular=is_regular, clustered_alignments=eval_dataset["clustered_alignments"], full_inputs=eval_dataset["full_inputs"], regular_highlights_inputs=eval_dataset["regular_highlights_inputs"], preprocessor=regular_highlights_preprocessor)

    # Initialize our Trainer
    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=curr_compute_metrics if training_args.predict_with_generate else None,
    )

    if (len(sys.argv) == 3 or len(sys.argv) == 2) and sys.argv[-1].endswith(".json") and (training_args.do_predict or not torch.distributed.is_initialized() or (torch.distributed.is_initialized() and torch.distributed.get_rank()==0)):
        config_file_path = sys.argv[-1]
        with open(config_file_path, "r") as f:
            config_file = json.loads(f.read())
        with open(f"{training_args.output_dir}/config_file.json", "w") as f:
            f.write(json.dumps(config_file))

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        while True:
            try:
                train_result = trainer.train(resume_from_checkpoint=checkpoint)
            except RuntimeError as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                if exc_type==torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    # Detecting last checkpoint.
                    last_checkpoint = get_last_checkpoint(training_args.output_dir)
                    if last_checkpoint==checkpoint: # there was no actual progress between two RuntimeError raises (or the RuntimeError raise was at the beginning of the training) - so the problem cannot be solves by simply emptying the cache
                        raise
                    checkpoint = last_checkpoint
                    continue
                else:
                    raise
            break
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(
                train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")
        metrics = trainer.evaluate(
            max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(
            eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            prep_predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(
                prep_predict_dataset)
        )
        metrics["predict_samples"] = min(
            max_predict_samples, len(prep_predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                PredictionsAnalyzer(tokenizer=tokenizer, 
                                    preprocessor=regular_highlights_preprocessor, 
                                    output_dir=training_args.output_dir, 
                                    rouge_metric=rouge_metric, 
                                    meteor_metric=meteor_metric).write_predictions_to_file(predict_results.predictions, prep_predict_dataset, pd.DataFrame(predict_dataset.to_dict()))

    kwargs = {"finetuned_from": model_args.model_name_or_path,
              "tasks": "summarization"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    if data_args.lang is not None:
        kwargs["language"] = data_args.lang

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
