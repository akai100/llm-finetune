from accelerate import Accelerator
from transformers import Trainer, TrainingArguments

def main():
    accelerator = Accelerator()

    model, tokenizer = load_model(cfg)
    train_ds = build_dataset()

    training_args = TrainingArguments(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.bs,
        gradient_accumulation_steps=cfg.grad_acc,
        fp16=cfg.fp16,
        logging_steps=10,
        save_steps=500,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        tokenizer=tokenizer
    )

    trainer.train()

if __name__ == "__main__":
    main()

