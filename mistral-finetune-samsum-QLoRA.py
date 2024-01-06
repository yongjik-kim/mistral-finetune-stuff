from accelerate import FullyShardedDataParallelPlugin, Accelerator
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    FullOptimStateDictConfig,
    FullStateDictConfig,
)
from datasets import load_dataset
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model, PeftModel
import transformers

train_dataset = load_dataset("samsum", split="train")
eval_dataset = load_dataset("samsum", split="validation")
test_dataset = load_dataset("samsum", split="test")

fsdp_plugin = FullyShardedDataParallelPlugin(
    state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
    optim_state_dict_config=FullOptimStateDictConfig(
        offload_to_cpu=True, rank0_only=False
    ),
)
accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    cache_dir="models",
)

base_model_id = "mistralai/Mistral-7B-v0.1"
model = AutoModelForCausalLM.from_pretrained(
    base_model_id, load_in_8bit=True, quantization_config=bnb_config
)
tokenizer = AutoTokenizer.from_pretrained(
    base_model_id, model_max_length=512, padding_side="left", add_eos_token=True
)
tokenizer.pad_token = tokenizer.eos_token


def tokenize(prompt):
    result = tokenizer(prompt, truncation=True, max_length=1024, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result


# I'm giving a very simple prompt to see fine-tuning effect more clearly.
# With good enough prompt, Mistral-7B base model should be able to do summarization already pretty well.
def generate_and_tokenize_prompt(data_point):
    full_prompt = f"""
### Dialogue:
{data_point["dialogue"]}

### Summary:
{data_point["summary"]}
"""
    return tokenize(full_prompt)


tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
        "lm_head",
    ],
    bias="none",
    lora_dropout=0.05,
    task_type="CAUSAL_LM",
)

model = accelerator.prepare_model(get_peft_model(model, config))


project = "samsum-finetune-LoRA"
base_model_name = "mistral"
run_name = base_model_name + "-" + project
output_dir = "./checkpoints/" + run_name
tokenizer.pad_token = tokenizer.eos_token

torch.cuda.empty_cache()

trainer = transformers.Trainer(
    model=model,
    train_dataset=tokenized_train_dataset,
    eval_dataset=tokenized_val_dataset,
    args=transformers.TrainingArguments(
        output_dir=output_dir,
        warmup_steps=5,
        per_device_train_batch_size=2,
        gradient_checkpointing=True,
        gradient_accumulation_steps=4,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_steps=2000,
        learning_rate=2.5e-5,  # Want about 10x smaller than the Mistral learning rate
        logging_steps=200,
        bf16=True,
        optim="paged_adamw_8bit",
        logging_dir="./logs",  # Directory for storing logs
        save_strategy="steps",  # Save the model checkpoint every logging step
        save_steps=200,  # Save checkpoints every 50 steps
        evaluation_strategy="steps",  # Evaluate the model every logging step
        eval_steps=200,  # Evaluate and save checkpoints every 50 steps
        do_eval=True,  # Perform evaluation at the end of training
    ),
    data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

model.config.use_cache = False  # silence the warnings. Please re-enable for inference!

trainer.train()
