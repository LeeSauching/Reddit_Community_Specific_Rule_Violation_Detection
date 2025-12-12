def load_and_expand() -> pd.DataFrame:
    comp_dir = CFG.comp_dir
    train_dataset = pd.read_csv(f"{comp_dir}/train.csv")
    test_dataset  = pd.read_csv(f"{comp_dir}/test.csv")

    # Base train rows
    base_cols = ["body", "rule", "subreddit", "rule_violation",
                 "positive_example_1", "positive_example_2",
                 "negative_example_1", "negative_example_2"]
    train_df = train_dataset[base_cols].copy()

    merged = []
    for violation_type in ["positive","negative"]:
        for i in [1,2]:
            body_col = f"{violation_type}_example_{i}"
            sub = test_dataset[["rule","subreddit",
                                "positive_example_1","positive_example_2",
                                "negative_example_1","negative_example_2"]].copy()
            sub["body"] = sub[body_col]
            sub["rule_violation"] = 1 if violation_type=="positive" else 0
            merged.append(sub[["body","rule","subreddit","rule_violation"]])

    plain = train_df[["body","rule","subreddit","rule_violation"]].copy()
    merged.append(plain)

    df = pd.concat(merged, axis=0, ignore_index=True)
    # Clean
    df["body"] = df["body"].fillna("").astype(str)
    df["rule"] = df["rule"].fillna("").astype(str)
    df["subreddit"] = df["subreddit"].fillna("").astype(str)
    df["rule_violation"] = df["rule_violation"].fillna(0).astype(int)

    # Drop empty bodies (sometimes example cells may be blank)
    df = df[df["body"].str.strip().ne("")]

    # Clean label
    df["_body_trim"] = df["body"].str.strip()
    df["_rule_trim"] = df["rule"].str.strip()
    grp = df.groupby(["_body_trim", "_rule_trim"], dropna=False)

    nunique_labels = grp["rule_violation"].nunique().reset_index(name="label_nunique")
    conflict_keys = nunique_labels[nunique_labels["label_nunique"] > 1][["_body_trim", "_rule_trim"]].drop_duplicates()

    if len(conflict_keys) > 0:
        stats = grp["rule_violation"].agg(["sum", "count"]).reset_index()
        stats["mode_label"] = (stats["sum"] > (stats["count"] / 2)).astype(int)

        df = df.merge(stats[["_body_trim", "_rule_trim", "mode_label"]], on=["_body_trim", "_rule_trim"], how="left")
        df = df.merge(conflict_keys.assign(_is_conflict=True), on=["_body_trim", "_rule_trim"], how="left")
        conflict_mask = df["_is_conflict"].fillna(False)
        df.loc[conflict_mask, "rule_violation"] = df.loc[conflict_mask, "mode_label"].astype(int)

        df = df.drop(columns=["mode_label", "_is_conflict"])

    df = df.drop(columns=["_body_trim", "_rule_trim"])

    # Deduplicate 
    if CFG.dd_method == "all":
        df["key"] = df["body"].str.strip() + "||" + df["rule"].str.strip() + "||" + df["subreddit"].str.strip() + "||" + df["rule_violation"].astype(str)
    elif CFG.dd_method == "no_subreddit":
        df["key"] = df["body"].str.strip() + "||" + df["rule"].str.strip() + "||" + df["rule_violation"].astype(str)

    df = df.drop_duplicates(subset=["key"]).drop(columns=["key"]).reset_index(drop=True)
    

    rule_counts = df.groupby("rule").size().to_dict()
    max_count = max(rule_counts.values()) if len(rule_counts) > 0 else 0
    upsampled_parts = [df]
    for rule_name, count in rule_counts.items():
        if count <= 0:
            continue
            
        factor = max_count / count
        factor_int = max_count // count

        if factor_int > CFG.max_upsample_ratio:
            factor_int = CFG.max_upsample_ratio
        
        rule_block = df[df["rule"] == rule_name]

        copies = []
        for _ in range(factor_int):
            if factor > 1.2:
                copies.append(rule_block.copy())
            else:
                copies.append(rule_block.copy()[:int(len(rule_block)*0.5)]) # Avoid timeout
                
        if len(copies) > 0:
            sampled_block = pd.concat(copies, axis=0, ignore_index=True)
        else:
            sampled_block = pd.DataFrame(columns=df.columns)

        
        upsampled_parts.append(sampled_block)

    df = pd.concat(upsampled_parts, axis=0, ignore_index=True)

    if CFG.debug:
        df = df.sample(min(500, len(df)), random_state=CFG.seed).reset_index(drop=True)

    # Helper column for supervised SFT
    df["rule_violation_str"] = df["rule_violation"].map({1: CFG.positive, 0: CFG.negative})
    return df, test_dataset

def train_val_split(df: pd.DataFrame, val_ratio=0.1, seed=42):
    if val_ratio is None or val_ratio <= 0:
        return df.reset_index(drop=True), df.iloc[0:0].copy()

    # Stratified by label
    n_splits = int(round(1.0 / val_ratio))
    n_splits = max(2, n_splits)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    y = df["rule_violation"].values
    idx = np.arange(len(df))
    train_idx, val_idx = next(iter(skf.split(idx, y)))
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)

def build_hf_dataset(df: pd.DataFrame, tokenizer, system_prompt: str, use_log: bool):
    convs = conversations_from_df(df, system_prompt, CFG.judge_words)
    text_list = tokenizer.apply_chat_template(
        convs, tokenize=False, # enable_thinking=False
        )
    
    if CFG.strip_reasoning_blocks:
        text_list = [strip_thinking_tags(t) for t in text_list]
            
    if use_log:
        LOGGER.info(f"=== Example Complete Chat Template ===\n{text_list[0]}\n{'='*50}\n")
    return HFDataset.from_dict({"text": text_list})
