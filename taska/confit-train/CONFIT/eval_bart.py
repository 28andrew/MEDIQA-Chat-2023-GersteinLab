#!/usr/bin/env python
# coding=utf-8
# Copyright 2021 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.

import logging
import os
import sys
from dataclasses import field, dataclass
from typing import Optional

import nltk  # Here to have a nice missing dependency error message early on
import numpy as np
import random
import torch
from datasets import load_metric

import transformers

from accelerate import Accelerator
from filelock import FileLock
from transformers import (
    DataCollatorForTokenClassification, 
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    HfArgumentParser,
    Seq2SeqTrainingArguments,
    get_scheduler,
    default_data_collator,
    Seq2SeqTrainer,
    # ModelArguments,
    # DataTrainingArguments,
    BartForConditionalGeneration,
    DataCollatorForSeq2Seq
)
from transformers.file_utils import is_offline_mode
from transformers.trainer_utils import get_last_checkpoint, is_main_process
from transformers.utils import check_min_version
from datasets import Dataset
import jsonlines
from transformers import BartConfig,PreTrainedTokenizerFast,BartTokenizer

from torch.utils.data.dataloader import DataLoader

import argparse

# parser = argparse.ArgumentParser(description='Process args.')
# parser.add_argument('--outdir', type=str, required=True)

# args = parser.parse_args()
# outdir= args.outdir
# print (outdir)


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
check_min_version("4.4.0.dev0")

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

alpha=0.01
model_name='linydub/bart-large-samsum'
# model_name = 'jackieliu930/bart-large-cnn-samsum'
# model_name = 'lidiya/bart-large-xsum-samsum'
# model_name = 'philschmid/distilbart-cnn-12-6-samsum'
# model_name = 'Salesforce/bart-large-xsum-samsum'
# outdir='output-' + model_name.split('/')[1] + '-large-epoch-2'
# model_name = 'lidiya/bart-base-samsum'
# model_name = 'facebook/bart-large'
# model_name = 'facebook/mbart-large-cc25'
# model_name = 'facebook/bart-large-xsum'

# model_name = 'facebook/bart-large'
outdir = '677-samsum-bart'
# outdir = 'output-' + model_name.split('/')[1] + '-new-baseline'
# outdir = 'output-' + model_name.split('/')[1] + '-baseline'

if not os.path.exists(outdir):
    os.mkdir(outdir)
dirs=os.listdir(outdir)
dir_idx=len(dirs)+1
dir_idx=1
print(dir_idx)
seed=str(dir_idx*1000)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """
    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    train_file: Optional[str] = field(
        default=None, metadata={"help": "The input training data file (a text file)."}
    )
    train_data_files: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "The input training data files (multiple files in glob format). "
                "Very often splitting large files to smaller files can prevent tokenizer going out of memory"
            )
        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."},
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input train ref data file for whole word mask in Chinese."},
    )
    eval_ref_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input eval ref data file for whole word mask in Chinese."},
    )
    line_by_line: bool = field(
        default=False,
        metadata={"help": "Whether distinct lines of text in the dataset are to be handled as distinct sequences."},
    )

    mlm: bool = field(
        default=False, metadata={"help": "Train with masked-language modeling loss instead of language modeling."}
    )
    whole_word_mask: bool = field(default=False, metadata={"help": "Whether ot not to use whole word mask."})
    mlm_probability: float = field(
        default=0.15, metadata={"help": "Ratio of tokens to mask for masked language modeling loss"}
    )
    plm_probability: float = field(
        default=1 / 6,
        metadata={
            "help": (
                "Ratio of length of a span of masked tokens to surrounding context length for permutation language"
                " modeling."
            )
        },
    )
    max_span_length: int = field(
        default=5, metadata={"help": "Maximum length of a span of masked tokens for permutation language modeling."}
    )

    block_size: int = field(
        default=-1,
        metadata={
            "help": (
                "Optional input sequence length after tokenization."
                "The training dataset will be truncated in block of this size for training."
                "Default to the model max input length for single sentence inputs (take into account special tokens)."
            )
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )

args=[
    # '--model_name_or_path','facebook/bart-large',
    # '--model_name_or_path','lidiya/bart-base-samsum',
    '--model_name_or_path', 'trained-model',
    '--do_predict',
    # '--do_train','--do_eval','--do_predict',
    # '--do_train','--do_predict',
    '--train_file','TaskA-TrainingData.json',
    '--validation_file','TaskA-TrainingData.json',
    '--test_file','TaskA-TestData.json',
    '--output_dir',outdir+'/'+str(dir_idx),
    '--per_device_train_batch_size=4',
    '--per_device_eval_batch_size=4',
    '--overwrite_output_dir',
    '--predict_with_generate=1',
    '--seed',seed,
    '--num_train_epochs','20',
    '--save_strategy','epoch',
    '--evaluation_strategy','epoch',
    '--learning_rate','1e-6',
    '--gradient_accumulation_steps=2',
    '--label_smoothing_factor=0.1',
    '--load_best_model_at_end',
    '--metric_for_best_model','eval_rouge1',
]
parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses(args)


# ARGS
data_args.max_source_length = 2048
data_args.max_target_length = 256
data_args.val_max_target_length = 256
data_args.ignore_pad_token_for_loss = True
data_args.num_beams = 4

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        
set_seed(training_args.seed)
training_args.remove_unused_columns=False

if 1:

    last_checkpoint=None

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )



    # Set the verbosity to info of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity_info()
    logger.info("Training/evaluation parameters %s", training_args)

    # Get the datasets: you can either provide your own CSV/JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For CSV/JSON files this script will use the first column for the full texts and the second column for the
    # summaries (unless you specify column names for this with the `text_column` and `summary_column` arguments).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
def load_jsonl(file_path):
    import itertools
    import json
    with jsonlines.open(file_path, "r") as rfd:
        result={}
        keys=['dialogue','summary','utterances']
        for key in keys:
            result[key]=[]
        for data in rfd:
            text=data['dialogue']
            text=list(map(lambda x:x.strip().split(' '),text))
            
            utterance_dic={}
            total_words=0 
            
            for utterance in text:
                try:
                    name = utterance[0]
                    addi_len = 0
                    if ":" not in name:
                        if utterance[1][-1] == ':':
                            name = utterance[0] + utterance[1]
                            addi_len = 1
                        elif utterance[2][-1] == ':':
                            name = utterance[0] + utterance[1] + utterance[2]
                            addi_len = 2
                    if len(utterance) > addi_len + 1:
                        if name not in utterance_dic.keys():
                            utterance_dic[name] = []
                        utterance_dic[name].append((total_words + 1 + addi_len, total_words + len(utterance) - 1))
                    else:
                        print(utterance)
                        pass
                    total_words += len(utterance)
                except:
                    pass
                
            text=list(itertools.chain(*text))
            # assert(len(text)==total_words)
            
            summary=data['summary']
            summary=list(map(lambda x:x.split(' '),summary))
            
            summary=list(itertools.chain(*summary))
            result['summary'].append(summary)
            result['dialogue'].append(text)
            result['utterances'].append(json.dumps(utterance_dic))
    return result

datasets={}
data_files = {}
if data_args.train_file is not None:
    data_files["train"] = data_args.train_file+'l'
    fil = data_files["train"]
    print(f"TRAIN FILE: {fil}")
    extension = data_args.train_file.split(".")[-1]
if data_args.validation_file is not None:
    data_files["validation"] = data_args.validation_file+'l'
    extension = data_args.validation_file.split(".")[-1]
if data_args.test_file is not None:
    data_files["test"] = data_args.test_file+'l'
    extension = data_args.test_file.split(".")[-1]
for key in data_files:
    print(key)
    dic=load_jsonl(data_files[key])
    datasets[key]=Dataset.from_dict(dic)

    # See more about loading any type of standard or custom dataset (from files, python dict, pandas DataFrame, etc) at
    # https://huggingface.co/docs/datasets/loading_datasets.html.

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    
config=BartConfig.from_pretrained(model_args.model_name_or_path)
model=BartForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
model.alpha=alpha
print(model.alpha)

tokenizer = AutoTokenizer.from_pretrained(
    model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
    cache_dir=model_args.cache_dir,
    use_fast=model_args.use_fast_tokenizer,
    revision=model_args.model_revision,
    use_auth_token=True if model_args.use_auth_token else None,
    #split into word
    add_prefix_space=True,
    padding=True,
    truncation=True
)
if 1:
    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    text_column='dialogue'
    summary_column='summary'
    label_column='utterances'
    column_names = datasets["train"].column_names
    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    # print (max_target_length)
    # raise Exception("STOP")

    padding = True

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warn(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    def preprocess_function(examples):
        inputs = examples[text_column]
        targets = examples[summary_column]
        MYlabels = examples[label_column]
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True,is_split_into_words=True)
        
        # print (model_inputs)
        # raise Exception("Stop here!")

        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=max_target_length, padding=padding, truncation=True,is_split_into_words=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        
        def find_head(l,x):
            return l.index(x)
        def find_tail(l,x):
            return len(l)-l[::-1].index(x)-1
        
        MYlabels_input=[]
        import json
        for i, label in enumerate(MYlabels):
            label=json.loads(label)
            word_ids = model_inputs.word_ids(batch_index=i)
            
            the_last=word_ids[-2]
            utterance={}
            for key in label.keys():
                utterance[key]=[]
                for span in label[key]:
                    if span[1]>the_last:
                        print(len(word_ids))
                        break
                    utterance[key].append([find_head(word_ids,span[0]),find_tail(word_ids,span[1])])
            MYlabels_input.append(json.dumps(utterance))
        model_inputs['speaker_label']=MYlabels_input
        return model_inputs

data_args.preprocessing_num_workers = 32

if 1:
    if training_args.do_train:
        train_dataset = datasets["train"]
        if "train" not in datasets:
            raise ValueError("--do_train requires a train dataset")
        if data_args.max_train_samples is not None:
            train_dataset = train_dataset.select(range(data_args.max_train_samples))
        train_dataset = train_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_eval:
        max_target_length = 128
        if "validation" not in datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_dataset = datasets["validation"]
        # if data_args.max_val_samples is not None:
        #     eval_dataset = eval_dataset.select(range(data_args.max_val_samples))
        eval_dataset = eval_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in datasets:
            raise ValueError("--do_predict requires a test dataset")
        test_dataset = datasets["test"]
        # if data_args.max_test_samples is not None:
        #     test_dataset = test_dataset.select(range(data_args.max_test_samples))
        test_dataset = test_dataset.map(
            preprocess_function,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )



# Data collator
label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

# Metric
metric = load_metric("rouge", cache_dir="./my_cache")

def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]

    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

    return preds, labels

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if data_args.ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)

    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    # Extract a few results from ROUGE
    result = {key: value.mid.fmeasure * 100 for key, value in result.items()}

    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    result = {k: round(v, 4) for k, v in result.items()}
    return result

# train_dataset.cleanup_cache_files()


trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )



# Training
if training_args.do_train:
    train_result = trainer.train(resume_from_checkpoint=None)
    trainer.save_model()  # Saves the tokenizer too for easy upload

    metrics = train_result.metrics
    max_train_samples = (
        data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
    )
    metrics["train_samples"] = min(max_train_samples, len(train_dataset))

    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

if training_args.do_predict:
    logger.info("*** Test ***")

    test_results = trainer.predict(
        test_dataset,
        metric_key_prefix="test",
        max_length=data_args.val_max_target_length,
        num_beams=data_args.num_beams,
    )
    metrics = test_results.metrics
    max_test_samples = data_args.max_test_samples if data_args.max_test_samples is not None else len(test_dataset)
    metrics["test_samples"] = min(max_test_samples, len(test_dataset))

    trainer.log_metrics("test", metrics)
    trainer.save_metrics("test", metrics)

    if trainer.is_world_process_zero():
        if training_args.predict_with_generate:
            test_preds = tokenizer.batch_decode(
                test_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            test_preds = [pred.strip() for pred in test_preds]
            output_test_preds_file = os.path.join(training_args.output_dir, "test_generations.txt")
            with open(output_test_preds_file, "w",encoding='UTF-8') as writer:
                writer.write("\n".join(test_preds))

print(model.alpha)
torch.save(model.total_speaker_loss,'total_speaker_loss')
torch.save(model.total_summary_loss,'total_summary_loss')
torch.save(model.total_loss,'total_loss')
