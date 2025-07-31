# Plant_Disease_analysis_writeup
This is a technical writeup for Gemma 3n impact Challenge
In order to fine tune gemma with plant disease dataset. I want this fine tune model to be a agent that specialize plant disease analyzes.

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
!pip install --no-deps --upgrade timm # Only for Gemma 3N
```



