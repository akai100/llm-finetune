from transformers import GenerationConfig

gen_cfg = GenerationConfig(
    max_new_tokens=256,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

outputs = model.generate(
    input_ids,
    generation_config=gen_cfg
)
