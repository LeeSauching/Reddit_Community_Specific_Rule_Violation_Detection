from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template

def make_lora_config() -> LoraConfig:
    return LoraConfig(
        r=CFG.lora_r,
        lora_alpha=CFG.lora_alpha,
        lora_dropout=CFG.lora_dropout,
        bias="none",
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )

#原本这部分散落在 main 函数里。建议封装成一个函数，返回 model 和 tokenizer。

#注意：记得在这里应用 chat_template。
def load_model_and_tokenizer(model_path):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name = model_path,
        max_seq_length = CFG.max_seq_len,
        load_in_4bit = True,
        load_in_8bit = False,
        full_finetuning = False,
        local_files_only= True,
    )
    print(f"tokenizer.pad_token: {tokenizer.pad_token}, tokenizer.eos_token: {tokenizer.eos_token}")
    print(f"tokenizer.pad_token_id: {tokenizer.pad_token_id}, tokenizer.eos_token_id: {tokenizer.eos_token_id}")

    tokenizer.truncation_side = "left" 
    tokenizer.padding_side    = "right"

    return model, tokenizer

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
