import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    default_data_collator,
)
from transformers.trainer_utils import get_last_checkpoint
from datasets import Dataset

# --------------------------------------------------
# Config
# --------------------------------------------------
MODEL_ID   = "gpt2"
DATA_FILE  = "data/train.txt"
OUTPUT_DIR = "gpt2_password_model"
MAX_LENGTH = 12    # 10 chars + 1 EOS + 1 buffer (each char = exactly 1 token now)
NUM_EPOCHS = 3
BATCH_SIZE = 512

# Set to None to train on the full dataset (~9.5M passwords, ~2-3 hrs)
# Set to a number (e.g. 50_000) for a fast smoke-test (~5-10 min)
SAMPLE_SIZE = None  # smoke-test. Change to None for full dataset (~9.5M passwords)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    print(f"Loading {MODEL_ID} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    model.config.pad_token_id = tokenizer.eos_token_id

    print(f"Loading passwords from {DATA_FILE} ...")
    with open(DATA_FILE, "r", encoding="utf-8", errors="ignore") as f:
        passwords = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(passwords):,} passwords")

    if SAMPLE_SIZE is not None:
        passwords = passwords[:SAMPLE_SIZE]
        print(f"Quick-test mode: using first {len(passwords):,} passwords")

    # Build char→token_id lookup once — avoids millions of tokenizer.encode() calls.
    all_chars = set(ch for p in passwords for ch in p)
    char_to_tid = {}
    for ch in all_chars:
        toks = tokenizer.encode(ch, add_special_tokens=False)
        if toks:
            char_to_tid[ch] = toks[0]
    EOS = tokenizer.eos_token_id
    PAD = tokenizer.pad_token_id

    # Tokenize directly (no dataset.map / no multiprocessing — avoids Windows pickle issues).
    print("Tokenizing dataset (char-level)...")
    all_input_ids, all_labels, all_masks = [], [], []
    for password in passwords:
        ids = [char_to_tid[ch] for ch in password if ch in char_to_tid]
        ids.append(EOS)
        ids = ids[:MAX_LENGTH]
        pad_len = MAX_LENGTH - len(ids)
        all_input_ids.append(ids + [PAD] * pad_len)
        all_labels.append(ids + [-100]  * pad_len)
        all_masks.append([1] * (MAX_LENGTH - pad_len) + [0] * pad_len)
    tokenized = Dataset.from_dict({
        "input_ids":      all_input_ids,
        "labels":         all_labels,
        "attention_mask": all_masks,
    })
    print("Tokenization complete.")

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        fp16=torch.cuda.is_available(),
        logging_steps=100,
        save_steps=5000,
        save_total_limit=2,
        report_to="none",
        dataloader_num_workers=0,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized,
        data_collator=default_data_collator,
    )

    last_ckpt = get_last_checkpoint(OUTPUT_DIR)
    if last_ckpt:
        print(f"Resuming from checkpoint: {last_ckpt}")
    trainer.train(resume_from_checkpoint=last_ckpt)

    print("Training complete! Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Weights saved to {OUTPUT_DIR}/")
