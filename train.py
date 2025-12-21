import argparse
import os
import torch
import yaml
import torch
from src.utils import load_config, seed_torch, init_logger
from src.data import load_and_expand, train_val_split, build_hf_dataset
from src.model import load_model_tokenizer, get_lora_model, get_trainer
from src.inference import evaluate_auc_on_val

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config")
    args = parser.parse_args()

	cfg = load_conig(args.config)

	df, _ = load_and_expand(
		comp_dir = cfg['data']['comp_dir'],
		dd_method = cfg['data']['dd_method'],
		max_upsample_ratio = cfg['data']['max_upsample_ratio'],
		seed = cfg['seed']
	)

	os.makedirs(cfg['output_dir'], exist_ok = True)

	logger.info(f"=== Starting Training: {cfg['model_name']} ===")

	logger.info("Loading and processing data...")

    seed_everything(cfg['seed'])
    logger = init_logger(f"{cfg['run_name'].log}")

    logger.info(f"Loading model: {cfg['model']['path']}")

    df, test_df = load_and_expand()
    train_df, val_df = train_val_split(df, cfg['data']['val_ratio'], cfg['seed'])
	logger.info(f"Train size: {len(train_df)}, Val size: {len(val_df)}")
	train_hf = build_hf_dataset(train_df, tokenizer, primary_system_prompt, use_log=True)
	val_hf   = build_hf_dataset(val_df,   tokenizer, primary_system_prompt, use_log=False) if len(val_df)>0 else None

	logger.info(f"Loading base model from: {cfg['base_model_path']}")
	model, tokenizer = load_model_and_tokenizer(cfg['base_model_path'], cfg['max_seq_len'])

	logger,info("Applying LoRA adapters...")
	model = get_lora_model(model, cfg)

	system_prompt = cfg["prompt_variants"]

	trainer = get_trainer(model, tokenizer, train_hf, vak_hf, cfg)

	logger.info("Starting training...")
	trainer_stats = trainer.train()

	if len(val_df) > 0:
		logger.info("Running evaluation...")
		auc = evaluate_auc_on_val(val_df, tokenizer, model, cfg["train"]["per_device_batch_size"]
		logger.info(f"Validation AUC: {auc:.6f}")

	logger.info(f"Saving model to {cfg["output_dir"]}")
    model.save_pretrained(cfg["output_dir"])
    tokenizer.save_pretrained(cfg["output_dir"])

if __name__ == "__main__":
    main()
	
