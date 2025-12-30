import torch
from transformers import (
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoTokenizer
)
from accelerate import Accelerator

from src.models.load_model import load_model
from data.dataset import InstructionDataset
from src.utils.config import Config
from data.preprocess import preprocess_dataset, train_eval_split
from src.training.watchdog import TrainingWatchdog

def main():
    cfg = Config([
        "configs/model.yaml",
        "configs/train.yaml",
        "configs/lora.yaml"
    ])
    cfg.override_from_cli()

    accelerator = Accelerator()

    model, tokenizer = load_model(cfg)

    raw_data = preprocess_dataset(cfg.train_file)
    train_data, eval_data = train_eval_split(raw_data)

    train_ds = InstructionDataset(train_data, tokenizer, cfg.max_len)
    eval_ds = InstructionDataset(eval_data, tokenizer, cfg.max_len)

    watchdog = TrainingWatchdog(timeout=900)  # 15 分钟
    watchdog.start()

    args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        num_train_epochs=cfg.num_train_epochs,
        fp16=cfg.fp16,
        bf16=cfg.bf16,
        logging_steps=cfg.logging_steps,
        save_steps=cfg.save_steps,
        evaluation_strategy="steps",
        eval_steps=cfg.eval_steps,
        save_total_limit=2,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        callbacks=[NaNLossCallback(), WatchdogCallback(watchdog)]
    )

    try:
        trainer.train(resume_from_checkpoint=cfg.resume_from)
    except RuntimeError as e:
        handled = handle_oom(e)
        if not handled:
            raise e
    finally:
        watchdog.stop()

if __name__ == "__main__":
    main()
