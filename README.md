# Plant_Disease_analysis_writeup
This is a technical writeup for Gemma 3n impact Challenge.

![Plant Disease scientific analyzer.JPG](https://github.com/tomtyiu/Plant_Disease_analysis_writeup/blob/main/Plant%20Disease%20scientific%20analyzer.JPG?raw=true)


To fine-tune Gemma with the plant disease dataset. I want this fine-tuned model to be an agent that specializes in plant disease analysis. Most models do not have accurate information on plant disease.  My purpose is to fine-tune the Gemma 3n model to specialize in scientific plant disease.

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





