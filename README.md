# Plant_Disease_analysis_writeup
This is a technical writeup for Gemma 3n impact Challenge.

![Plant Disease scientific analyzer.JPG](https://github.com/tomtyiu/Plant_Disease_analysis_writeup/blob/main/Plant%20Disease%20scientific%20analyzer.JPG?raw=true)


To fine-tune Gemma with the plant disease dataset. I want this fine-tuned model to be an agent that specializes in plant disease scientific analysis. Most models do not have accurate information on plant disease.  My purpose is to fine-tune the Gemma 3n model to specialize in scientific plant disease.

First I need to install the unsloth dependencies
```
%%capture
import os
if "COLAB_" not in "".join(os.environ.keys()):
    !pip install unsloth
else:
    # Do this only in Colab notebooks! Otherwise use pip install unsloth
    !pip install --no-deps bitsandbytes accelerate xformers==0.0.29.post3 peft trl triton cut_cross_entropy unsloth_zoo
    !pip install sentencepiece protobuf "datasets>=3.4.1,<4.0.0" "huggingface_hub>=0.34.0" hf_transfer
    !pip install --no-deps unsloth

%%capture
# Install latest transformers for Gemma 3N
!pip install --no-deps --upgrade timm # LOnly for Gemma 3N
```
We need to add LoRA adapters for parameter for fine tuning the model, allowing unsloth to train only 1% of all model parameter effecrtively.  When r is larger, it menas the higher the accuracy.   In order to fine tune vision latyer, we set to True.
```
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers

    r = 32,                           # The larger, the higher the accuracy, but might overfit
    lora_alpha = 32,                  # Recommended alpha == r at least
    lora_dropout = 0,
    bias = "none",
    random_state = 3407,
    use_rslora = False,               # We support rank stabilized LoRA
    loftq_config = None,               # And LoftQ
    target_modules = "all-linear",    # Optional now! Can specify a list if needed
    modules_to_save=[
        "lm_head",
        "embed_tokens",
    ],
)
```

We need to load the dataset.  I used minhhungg/plant-disease-dataset in Huggingface.  It has answers that is comprehensive and scientific.

Data Prep
```
from datasets import load_dataset
dataset = load_dataset("minhhungg/plant-disease-dataset", split = "train")
```

Than format the data
```
instruction = "Analyze the plant disease for this image."

def convert_to_conversation(sample):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": instruction},
                {"type": "image", "image": sample["image"]},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": sample["answer"]}]},
    ]
    return {"messages": conversation}
pass
```

I use supervised fine=tuning techniques to train the model with the plant-disease dataset.  The sft fine-tuning was 60 steps with learning_rate of 2e-4.

```
from unsloth.trainer import UnslothVisionDataCollator
from trl import SFTTrainer, SFTConfig

FastVisionModel.for_training(model) # Enable for training!

trainer = SFTTrainer(
    model=model,
    train_dataset=converted_dataset,
    processing_class=processor.tokenizer,
    data_collator=UnslothVisionDataCollator(model, processor),
    args = SFTConfig(
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4,
        gradient_checkpointing = True,

        # use reentrant checkpointing
        gradient_checkpointing_kwargs = {"use_reentrant": False},
        max_grad_norm = 0.3,              # max gradient norm based on QLoRA paper
        warmup_ratio = 0.03,
        max_steps = 60,
        #num_train_epochs = 2,          # Set this instead of max_steps for full training runs
        learning_rate = 2e-4,
        logging_steps = 1,
        save_strategy="steps",
        optim = "adamw_torch_fused",
        weight_decay = 0.01,
        lr_scheduler_type = "cosine",
        seed = 3407,
        output_dir = "outputs",
        report_to = "none",             # For Weights and Biases

        # You MUST put the below items for vision finetuning:
        remove_unused_columns = False,
        dataset_text_field = "",
        dataset_kwargs = {"skip_prepare_dataset": True},
        max_length = 2048,
    )
)
trainer_stats = trainer.train()
```

After the training, the Gemma PD model provide comprehensive analysis of the plant disease for given picture.

Here is example output of the model:
```
Coffee leaf spot is caused by the fungus Cercospora coffeicola. The disease is widespread in coffee-growing regions of the world and can cause significant yield losses.

Symptoms
Coffee leaf spot appears as small, circular to oval, dark brown to black spots on the upper leaf surface. These spots expand and coalesce, causing large areas of leaf discoloration. The leaves then turn brown and dry, resulting in reduced photosynthetic activity and yield.

Causes
The disease is caused by the fungus Cercospora coffeicola, which overwinters on fallen leaves and branches. It produces brown to black, fruiting bodies (perithecia) on the leaf surfaces. These bodies release infective spores (zoospores) that are carried by wind or rain to new hosts.

Management
Management focuses on reducing the fungus's overwintering stages:

* Remove fallen leaves and debris to reduce the fungus's overwintering sites.
* Use certified coffee planting materials to avoid introduction of new strains of the fungus.
* Apply protectant fungicides, such as chlorothalonil or thiophanate-methyl, at 2-4 week intervals during wet springs and warm temperatures (65-80°F, 18-26°C). Apply at a rate of 1-2 lbs/acre (1.1-2.3 kg/ha) in a spray mixture containing a wetting agent to maximize coverage.

Cultural Practices
Cultural practices that increase the plant's resistance to leaf spot can be implemented:

* Maintain good cultural practices, such as fertilizing, irrigation, and pest control, to support overall plant health.
* Plant resistant coffee varieties, such as those with enhanced water-use efficiency.
* Maintain adequate tree spacing and pruning to promote air circulation and sunlight penetration to lower leaf surfaces.

Long-term Solutions
Developing resistant coffee varieties, improving irrigation, and implementing crop rotation are long-term solutions to minimize disease risk. Research on the epidemiology of leaf spot and the development of new fungicides or biological control agents is ongoing.
```
## Challenges
The difficulty with this project is that I have to find the good quality dataset and also find ways to train the Gemma E4B.  It was difficult because there was no way to train Gemma E4B in the beginning for images.  
I asked [unsloth](https://docs.unsloth.ai/) to make a Vision fine tune notebook in order for me to be able to fine tune the model for plant disease project. 

## Model
Link to the model
- [PD_gemma-3n-E4B-v2](https://huggingface.co/EpistemeAI/PD_gemma-3n-E4B-v2)
- [PD_gemma-3n-E2B](https://huggingface.co/EpistemeAI/PD_gemma-3n-E2B)


## Demo
[Demo](https://huggingface.co/spaces/legolasyiu/Gemma3N-challenge)

## Special thanks
<div>
  <div style="display: flex; gap: 5px; align-items: center; ">
    <a href="https://github.com/unslothai/unsloth/">
      <img src="https://github.com/unslothai/unsloth/raw/main/images/unsloth%20new%20logo.png" width="133">
    </a>
  </div>
</div>

Thank for minhhungg for allow me to fine tune the Huggingface dataset:  minhhungg/plant-disease-dataset

## LICENSE

<a href="https://github.com/tomtyiu/Plant_Disease_analysis_writeup">Plant_Disease_analysis_writeup</a> © 2025 by <a href="https://creativecommons.org">Thomas Yiu</a> is licensed under 
<a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>
<img src="https://mirrors.creativecommons.org/presskit/icons/cc.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
<img src="https://mirrors.creativecommons.org/presskit/icons/by.svg" alt="" style="max-width: 1em;max-height:1em;margin-left: .2em;">
