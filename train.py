import argparse
import os
import torch
import torch
from src.utils import load_config, seed_everything, init_logger
from src.data import load_and_expand, train_val_split, build_hf_dataset
from src.model import load_model_and_tokenizer, get_lora_model, get_trainer
from src.inference import evaluate_auc_on_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

    cfg = load_config(args.config)

    log_file = os.path.join(cfg.output_dir, f"train_{cfg.model_name}.log")
    os.makedirs(cfg.output_dir, exist_ok=True)
    logger = init_logger(log_file)
    logger.info(f"=== Starting Training: {cfg.model_name} ===")

    seed_everything(cfg.seed)

    logger.info("Loading and processing data...")

    df, test_df = load_and_expand(cfg)

    train_df, val_df = train_val_split(df, cfg.val_ratio, cfg.seed)
    logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")

    sys_prompt = cfg.prompt_variants[cfg.data.system_prompt_id]

    train_hf = build_hf_dataset(train_df, tokenizer, sys_prompt, use_log=True, cfg = cfg)
    val_hf   = build_hf_dataset(val_df,   tokenizer, sys_prompt, use_log=False, cfg = cfg) if len(val_df) > 0 else None

    logger.info(f"Loading base model from: {cfg['base_model_path']}")
    model, tokenizer = load_model_and_tokenizer(cfg['base_model_path'], cfg['max_seq_len'])

    logger.info("Applying LoRA adapters...")
    model = get_lora_model(model, cfg)

    trainer = get_trainer(model, tokenizer, train_hf, val_hf, cfg)

    logger.info("Starting training...")
    trainer_stats = trainer.train()

    if len(val_df) > 0:
        logger.info("Running evaluation...")
        auc = evaluate_auc_on_val(val_df, tokenizer, model, cfg)
        logger.info(f"Validation AUC: {auc:.6f}")

    logger.info(f"Saving model to {cfg["output_dir"]}")
    model.save_pretrained(cfg.output_dir)
    tokenizer.save_pretrained(cfg.output_dir)

if __name__ == "__main__":
    main()
