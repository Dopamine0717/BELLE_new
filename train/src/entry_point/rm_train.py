# coding=utf-8
from dataclasses import dataclass, field
from functools import partial
import math
import os
import sys
from typing import Any, Dict, List, Optional, Union
import torch

from accelerate import Accelerator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from tqdm import tqdm
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    BitsAndBytesConfig,
    HfArgumentParser,
    PreTrainedTokenizerBase,
    LlamaTokenizer,
    TrainingArguments,
    AutoModelForCausalLM
)
from transformers.utils import PaddingStrategy
from transformers.trainer_utils import get_last_checkpoint
from trl import RewardConfig, RewardTrainer
from trl.trainer.utils import RewardDataCollatorWithPadding
import logging
from multiprocessing import cpu_count
import subprocess
from transformers.utils import add_start_docstrings

# tqdm.pandas()
# accelerator = Accelerator()
# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def print_rank_0(msg, log_file, rank=0):
    if rank <= 0:
        with open(log_file, "a") as f:
            print(msg)
            f.write(msg + "\n")


@dataclass
# class ScriptArguments:
@add_start_docstrings(TrainingArguments.__doc__)
class TrainingArguments(TrainingArguments):
    """
    Hyperparameters to fine-tune a reward model on a given dataset with the `RewardTrainer`.
    """

    # Training arguments
    report_to: Optional[str] = field(
        default=None, metadata={"help": "use 'wandb' to log with wandb"}
    )
    logging_steps: Optional[int] = field(
        default=500, metadata={"help": "the number of update steps between two logs"}
    )
    learning_rate: Optional[float] = field(
        default=1.41e-5, metadata={"help": "the learning rate"}
    )
    per_device_train_batch_size: Optional[int] = field(
        default=1, metadata={"help": "the batch size"}
    )
    per_device_eval_batch_size: Optional[int] = field(
        default=1, metadata={"help": "evaluating batch size"}
    )
    num_train_epochs: Optional[int] = field(
        default=1, metadata={"help": "the number of training epochs"}
    )
    gradient_accumulation_steps: Optional[int] = field(
        default=16, metadata={"help": "the number of gradient accumulation steps"}
    )
    gradient_checkpointing: Optional[bool] = field(
        default=True, metadata={"help": "Enable gradient checkpointing"}
    )
    output_dir: Optional[str] = field(
        default="output", metadata={"help": "the output directory"}
    )
    fp16: Optional[bool] = field(default=False, metadata={"help": "float16"})
    bf16: Optional[bool] = field(default=False, metadata={"help": "bfloat16"})
    lr_scheduler_type: Optional[str] = field(
        default="linear",
        metadata={"help": "The lr scheduler"},
    )
    weight_decay: float = field(
        default=0.001, metadata={"help": "Weight decay for AdamW if we apply some."}
    )
    warmup_steps: int = field(
        default=1000, metadata={"help": "Linear warmup over warmup_steps."}
    )
    overwrite_output_dir: bool = field(
        default=False,
        metadata={
            "help": (
                "Overwrite the content of the output directory. "
                "Use this to continue training if output_dir points to a checkpoint directory."
            )
        },
    )
    resume_from_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": "The path to a folder with a valid checkpoint for your model."
        },
    )
    seed: int = field(
        default=42,
        metadata={"help": "Random seed that will be set at the beginning of training."},
    )
    dataloader_drop_last: Optional[bool] = field(
        default=True,
        metadata={
            "help": "Drop the last incomplete batch if it is not divisible by the batch size."
        },
    )
    save_total_limit: Optional[int] = field(
        default=3,
        metadata={
            "help": (
                "If a value is passed, will limit the total amount of checkpoints. Deletes the older checkpoints in"
                " `output_dir`. When `load_best_model_at_end` is enabled, the 'best' checkpoint according to"
                " `metric_for_best_model` will always be retained in addition to the most recent ones. For example,"
                " for `save_total_limit=5` and `load_best_model_at_end=True`, the four last checkpoints will always be"
                " retained alongside the best model. When `save_total_limit=1` and `load_best_model_at_end=True`,"
                " it is possible that two checkpoints are saved: the last one and the best one (if they are different)."
                " Default is unlimited checkpoints"
            )
        },
    )
    # Other arguments
    model_name: Optional[str] = field(
        default="facebook/opt-350m", metadata={"help": "the model name"}
    )
    train_data: str = field(default="", metadata={"help": "train data path"})
    eval_data: str = field(default="", metadata={"help": "eval data path"})
    cache_dir: str = field(default="", metadata={"help": "cache dir"})
    use_llama: Optional[bool] = field(default=False, metadata={"help": "bfloat16"})
    load_in_8bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 8 bits precision"}
    )
    load_in_4bit: Optional[bool] = field(
        default=False, metadata={"help": "load the model in 4 bits precision"}
    )
    use_lora: Optional[bool] = field(
        default=False, metadata={"help": "Wether to use LoRA or not to train adapters"}
    )
    trust_remote_code: Optional[bool] = field(
        default=True, metadata={"help": "Enable `trust_remote_code`"}
    )
    seq_length: Optional[int] = field(
        default=512, metadata={"help": "Input sequence length"}
    )
    deepspeed: str = field(
        default=None,
        metadata={
            "help": (
                "Enable deepspeed and pass the path to deepspeed json config file (e.g. `ds_config.json`) or an already"
                " loaded json file as a dict"
            )
        },
    )

def init_slurm_env():
    if 'SLURM_PROCID' in os.environ:
        proc_id = int(os.environ['SLURM_PROCID'])
        if proc_id==0:
            print('Init dist using slurm!')
            print("Job Id is {} on {} ".format(os.environ["SLURM_JOBID"], os.environ['SLURM_NODELIST']))

        ntasks = int(os.environ['SLURM_NTASKS'])
        # node_list = os.environ['SLURM_NODELIST']
        node_list = os.environ['SLURM_STEP_NODELIST']
        # node_list = os.environ['SLURM_STEP_NODELIST']
        num_gpus = torch.cuda.device_count()
        addr = subprocess.getoutput(
            'scontrol show hostname {} | head -n1'.format(node_list))
        jobid = os.environ["SLURM_JOBID"]
        stepid = os.environ["SLURM_STEP_ID"]
       

        tcp_port = os.environ.get('MASTER_PORT', 9904)


        os.environ['MASTER_PORT'] =str(tcp_port)
        os.environ['MASTER_ADDR'] = addr
        os.environ['WORLD_SIZE'] = str(ntasks)
        os.environ['RANK'] = str(proc_id)
        os.environ['LOCAL_RANK'] = str(proc_id % num_gpus)
        os.environ['LOCAL_SIZE'] = str(num_gpus)

        print('rank: {} world size: {} addr: {}  port: {}'.format(proc_id, ntasks, addr, os.environ['MASTER_PORT']))

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


# Tokenize chosen/rejected pairs of inputs
# Adapt this section to your needs for custom datasets
def preprocess_function(tokenizer: PreTrainedTokenizerBase, examples: Dict[str, Any]):
    new_examples = {
        "input_ids_chosen": [],
        "attention_mask_chosen": [],
        "input_ids_rejected": [],
        "attention_mask_rejected": [],
    }
    # for chosen, rejected in zip(examples["chosen"], examples["rejected"]):
    chosen = '<|im_start|>Human:\n' + examples['question'] + '<|im_end|>\n<|im_start|>Assistant:\n' + examples['answers'][0]['query'] + '<|im_end|>'
    rejected = '<|im_start|>Human:\n' + examples['question'] + '<|im_end|>\n<|im_start|>Assistant:\n' + examples['answers'][1]['query'] + '<|im_end|>'
    tokenized_chosen = tokenizer(chosen, add_special_tokens=False)
    tokenized_rejected = tokenizer(rejected, add_special_tokens=False)

    new_examples["input_ids_chosen"] = tokenized_chosen["input_ids"]
    new_examples["attention_mask_chosen"] = tokenized_chosen["attention_mask"]
    new_examples["input_ids_rejected"] = tokenized_rejected["input_ids"]
    new_examples["attention_mask_rejected"] = tokenized_rejected["attention_mask"]

    return new_examples

def main():
    init_slurm_env()
    parser = HfArgumentParser(TrainingArguments)
    script_args = parser.parse_args_into_dataclasses()[0]
    log_file = os.path.join(script_args.output_dir, "print_log.txt")
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    global_rank = torch.distributed.get_rank()
    num_gpus = torch.cuda.device_count()

    # Load the dataset and pre-process it
    if script_args.use_llama:
        tokenizer = LlamaTokenizer.from_pretrained(script_args.model_name)
        tokenizer.add_special_tokens(
            {
                "bos_token": "<s>",
                "eos_token": "</s>",
                "unk_token": "<unk>",
                "pad_token": "<unk>",
            }
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(script_args.model_name)
        # tokenizer.add_special_tokens({"pad_token": tokenizer.unk_token})
    tokenizer.padding_side = "left"
    print_rank_0(
        f"unk token: {tokenizer.unk_token}, "
        f"unk token id: {tokenizer.unk_token_id}, "
        f"pad token: {tokenizer.pad_token}, "
        f"pad token id: {tokenizer.pad_token_id}",
        log_file,
        global_rank
    )

    # with accelerator.main_process_first():
    train_dataset = load_dataset("json", data_files=script_args.train_data)["train"]
    eval_dataset = load_dataset("json", data_files=script_args.eval_data)["train"]

    # Preprocess the dataset and filter out examples that are longer than script_args.max_length
    train_dataset = train_dataset.map(
        partial(preprocess_function, tokenizer),
        # batched=True,
        # num_proc=max(cpu_count() // 2, 1),
        # remove_columns=["chosen", "rejected"],
    )
    train_dataset = train_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )

    eval_dataset = eval_dataset.map(
        partial(preprocess_function, tokenizer),
        # batched=True,
        # num_proc=max(cpu_count() // 2, 1),
        # remove_columns=["chosen", "rejected"],
    )
    eval_dataset = eval_dataset.filter(
        lambda x: len(x["input_ids_chosen"]) <= script_args.seq_length
        and len(x["input_ids_rejected"]) <= script_args.seq_length
    )

    for i in range(2):
        print_rank_0("Eval tokenized example: {}".format(train_dataset[i]), log_file, global_rank)
    for i in range(2):
        print_rank_0("Train tokenized example: {}".format(eval_dataset[i]), log_file, global_rank)

    # Define the training arguments
    training_nums = len(train_dataset)
    global_batch_size = num_gpus * script_args.gradient_accumulation_steps * script_args.per_device_train_batch_size
    if script_args.dataloader_drop_last:
        num_steps = (
            math.floor(training_nums / global_batch_size) * script_args.num_train_epochs
        )
    else:
        num_steps = (
            math.ceil(training_nums / global_batch_size) * script_args.num_train_epochs
        )
    eval_steps = max(num_steps // (script_args.num_train_epochs * 4), 5)
    print_rank_0(
        "num_gpus = {}, training_nums = {}, num_steps = {}, warmup_steps = {}, eval_steps = {}, save_steps = {}".format(
            num_gpus,
            training_nums,
            num_steps,
            script_args.warmup_steps,
            eval_steps,
            eval_steps,
        ),
        log_file,
        global_rank
    )
    # `TrainingArguments` must be instantiated before loading model!!!
    training_args = RewardConfig(
        output_dir=script_args.output_dir,
        per_device_train_batch_size=script_args.per_device_train_batch_size,
        per_device_eval_batch_size=script_args.per_device_eval_batch_size,
        num_train_epochs=script_args.num_train_epochs,
        gradient_accumulation_steps=script_args.gradient_accumulation_steps,
        gradient_checkpointing=script_args.gradient_checkpointing,
        learning_rate=script_args.learning_rate,
        report_to="tensorboard",
        remove_unused_columns=False,
        optim="adamw_torch",
        logging_steps=script_args.logging_steps,
        evaluation_strategy="steps",
        save_strategy="steps",
        max_length=script_args.seq_length,
        bf16=script_args.bf16,
        fp16=script_args.fp16,
        weight_decay=script_args.weight_decay,
        lr_scheduler_type=script_args.lr_scheduler_type,
        eval_steps=eval_steps,
        save_steps=eval_steps,
        warmup_steps=script_args.warmup_steps,
        overwrite_output_dir=script_args.overwrite_output_dir,
        resume_from_checkpoint=script_args.resume_from_checkpoint,
        save_total_limit=script_args.save_total_limit,
        load_best_model_at_end=True,
        ddp_timeout=3600,
        seed=script_args.seed,
        dataloader_drop_last=script_args.dataloader_drop_last,
        deepspeed=script_args.deepspeed,
    )

    print_rank_0("world_size = {}".format(training_args.world_size), log_file, global_rank)

    # Load the model
    if script_args.load_in_8bit and script_args.load_in_4bit:
        raise ValueError(
            "You can't load the model in 8 bits and 4 bits at the same time"
        )
    elif script_args.load_in_8bit or script_args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_8bit=script_args.load_in_8bit, load_in_4bit=script_args.load_in_4bit
        )
        # Copy the model to each device
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if world_size != 1 else "auto"
    else:
        device_map = None
        quantization_config = None

    # Model must be loaded after create `TrainingArguments`!!!
    model = AutoModelForSequenceClassification.from_pretrained(
        script_args.model_name,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=script_args.trust_remote_code,
        num_labels=1,
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    # model = AutoModelForCausalLM.from_pretrained(
    #             script_args.model_name,
    #             # torch_dtype=torch_dtype,
    #         )

    # Define the LoraConfig
    if script_args.use_lora:
        peft_config = LoraConfig(
            r=16,
            lora_alpha=16,
            bias="none",
            task_type="SEQ_CLS",
            modules_to_save=["scores"],
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()

    # Define the Trainer
    model.config.use_cache = False
    trainer = RewardTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=RewardDataCollatorWithPadding(
            tokenizer=tokenizer, pad_to_multiple_of=8
        ),
    )

    # Detecting last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint
    trainer.train(resume_from_checkpoint=checkpoint)
    trainer.save_model(training_args.output_dir)
    # accelerator.wait_for_everyone()
    print_rank_0("\n Training completed!!! If there's a warning about missing keys above, please disregard :)", log_file, global_rank)


if __name__ == "__main__":
    main()
