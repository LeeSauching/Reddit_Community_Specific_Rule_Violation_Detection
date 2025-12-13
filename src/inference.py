def yes_prob_from_next_token_logits(full_text_template, tokenizer, model, max_len=CFG.max_seq_len, bs=4):
    device = torch.device("cuda:0")
    FastLanguageModel.for_inference(model)
    model.eval()

    yes_id = tokenizer.encode("Yes", add_special_tokens=False)[0]  # int
    no_id  = tokenizer.encode("No", add_special_tokens=False)[0]   # int

    probs_all = []

    for i in range(0, len(full_text_template), bs):
        chunk = full_text_template[i:i+bs]

        enc = tokenizer(
            chunk,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_len,
            add_special_tokens=False,
        )

        input_ids = enc["input_ids"].to(device)          # (B, T)
        attention_mask = enc["attention_mask"].to(device)  # (B, T)

        am = attention_mask.to(dtype=torch.long)         # (B, T)
        position_ids = am.cumsum(dim=1) - 1              # padding -> -1
        position_ids.clamp_(min=0)                       # clamp -> 0
        position_ids = position_ids.to(device)

        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=True,
        )

        logits_last = outputs.logits[:, -1, :]           # (B, V)
        log_probs = torch.log_softmax(logits_last, dim=-1)  # (B, V)

        # get log probs for Yes/No first token
        log_yes = log_probs[:, yes_id]                   # (B,)
        log_no  = log_probs[:,  no_id]                   # (B,)

        denom = torch.logaddexp(log_yes, log_no)         # (B,)
        p_yes = torch.exp(log_yes - denom).clamp(0, 1)   # (B,)

        probs_all.append(p_yes.detach().cpu().numpy())

        del enc, input_ids, attention_mask, outputs
        torch.cuda.empty_cache()

    return np.concatenate(probs_all, axis=0)

def evaluate_auc_on_val(val_df: pd.DataFrame, tokenizer, model) -> float:
    texts = make_texts_for_variant(val_df, tokenizer, CFG.prompt_variants[CFG.pid], use_log=True)
    probs = yes_prob_from_next_token_logits(texts, tokenizer, model, bs=CFG.per_device_train_batch_size)
    auc = roc_auc_score(val_df["rule_violation"].values, probs)
    return float(auc)

