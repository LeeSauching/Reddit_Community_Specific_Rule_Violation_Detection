from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template
from trl import SFTTrainer, SFTconfig

def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )

def load_model_and_tokenizer(model_path, max_seq_len, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        moadel_name = model_path,
        max_seq_length = max_seq_len,
        dtype = None,                 # None = 自动检测 (Float16/Bfloat16)
        load_in_4bit = load_in_4bit,
        local_files_only = True
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = 'chatml',
        mapping = {"role": "role", "content": "content", "user": "user", "assistant": "assistant"},
    )

    tokenizer.truncation_side = "Left"
    tokenizer.padding_side = "right"

    print(f"Loaded model from {model_path}")
    print(f"tokenizer.pad_token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"tokenizer.eos_token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    return model, tokenizer

def get_lora_model(model, cfg):
    model = FastLanguageModel.get_peft_model(
        model,
        r = cfg.lora.r,
        target_modules = cfg.lora.target_modules,
        lora_alpha = cfg.lora.alpha,
        lora_dropout = cfg.lora.dropout,
        bias = "none",
        use_gradient_checkpointing = "unsloth",
        random_state = cfg.seed,
    )
    model.print_trainable_parameters()

    return model

def get_trainer(model, tokenizer, train_dataset, val_dataset, cfg):
    args = SFTconfig(
        output_dir = cfg.output_dir,
        num_train_epochs = cfg.train.epochs,
        per_device_train_batch_size = cfg.train.per_device_batch_size,
        gradient_accumulation_steps = cfg.train.grad_accum_steps,
        learning_rate = float(cfg.train.lr),
        fp16 = not cfg.train.bf16,
        bf16 = cfg.train.bf16,
        logging_steps = 10,

        eval_strategy = "epoch" if val_dataset else "no",
        save_strategy = "epoch" if val_dataset else "no",
        save_total_limit = 1,

        max_seq_length = cfg.max_seq_len,
        dataset_text_field = "text",
        packing = False,
        report_to = "none"
    )

    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<im_start>user\n",
        reponse_part = "<im_start>assistant\n"
    )
    
    return trainer
