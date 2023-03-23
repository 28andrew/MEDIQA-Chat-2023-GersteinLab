from dataclasses import dataclass
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel

import itertools
import random
import torch
import json
@dataclass
class DataCollatorForSeq2Seq:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        labels = [feature["labels"] for feature in features] if "labels" in features[0].keys() else None
        # We have to pad the labels before calling `tokenizer.pad` as this method won't pad them and needs them of the
        # same length to return tensors.
        if labels is not None:
            max_label_length = max(len(l) for l in labels)
            padding_side = self.tokenizer.padding_side
            for feature in features:
                remainder = [self.label_pad_token_id] * (max_label_length - len(feature["labels"]))
                feature["labels"] = (
                    feature["labels"] + remainder if padding_side == "right" else remainder + feature["labels"]
                )
        
        #speaker
        speaker_names=[]
        for feature in features:
            speaker_names.append(feature.pop('speaker_label'))

        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )

        # prepare decoder_input_ids
        if self.model is not None and hasattr(self.model, "prepare_decoder_input_ids_from_labels"):
            decoder_input_ids = self.model.prepare_decoder_input_ids_from_labels(labels=features["labels"])
            features["decoder_input_ids"] = decoder_input_ids
            
        #speaker
        global_idxs=[]
        pos_idxs=[]
        neg_idxs=[]
        max_len=features['input_ids'].size()[-1]
        
        try:
            for speaker_name in speaker_names:
                speaker_name = json.loads(speaker_name)
                pos_idx = [0] * max_len
                global_idx = [0] * max_len
                neg_idx = [0] * max_len
                quality_names = []
                for key in speaker_name:
                    if len(speaker_name[key]) > 1:
                        quality_names.append(key)

                if len(quality_names) == 0:

                    global_name, neg_name = random.sample(speaker_name.keys(), 2)

                    global_idx_range = random.sample(speaker_name[global_name], 1)[0]
                    for idx in range(global_idx_range[0], global_idx_range[1] + 1):
                        global_idx[idx] = 1

                    neg_idx_range = random.sample(speaker_name[neg_name], 1)[0]
                    for idx in range(neg_idx_range[0], neg_idx_range[1] + 1):
                        neg_idx[idx] = 1
                else:
                    global_name = random.sample(quality_names, 1)[0]
                    names = list(speaker_name.keys())
                    names.remove(global_name)
                    neg_name = random.sample(names, 1)[0]

                    global_idx_ranges = speaker_name[global_name]
                    pos_idx_range = random.sample(global_idx_ranges, 1)[0]
                    global_idx_ranges.remove(pos_idx_range)

                    for global_idx_range in global_idx_ranges:
                        for idx in range(global_idx_range[0], global_idx_range[1] + 1):
                            global_idx[idx] = 1

                    for idx in range(pos_idx_range[0], pos_idx_range[1] + 1):
                        pos_idx[idx] = 1

                    neg_idx_range = random.sample(speaker_name[neg_name], 1)[0]
                    for idx in range(neg_idx_range[0], neg_idx_range[1] + 1):
                        neg_idx[idx] = 1
                global_idxs.append(global_idx)
                neg_idxs.append(neg_idx)
                pos_idxs.append(pos_idx)
        except:
            pass
            
        features['global_idxs']=torch.ByteTensor(global_idxs).bool()
        features['pos_idxs']=torch.ByteTensor(pos_idxs).bool()
        features['neg_idxs']=torch.ByteTensor(neg_idxs).bool()

        return features