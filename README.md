# Video Analysis with Qwen2-VL-2B-Instruct

This script utilizes the Qwen2-VL-2B-Instruct model to process a video and provide a natural language understanding of the content, such as identifying the number of doctors in a video and describing their actions.

## Prerequisites

Install the necessary libraries before running the script:

```bash
!pip install git+https://github.com/huggingface/transformers accelerate flash_attn
!pip install qwen_vl_utils av
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import torch

# Define model name
model_name = "Qwen/Qwen2-VL-2B-Instruct"

# Input video and query
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "video",
                "video": "/content/3197622-hd_1920_1080_25fps.mp4",
                "max_pixels": 360 * 420,
                "fps": 1.0,
            },
            {"type": "text", "text": "How many doctors are there and what they are doing?"},
        ],
    }
]

# Process text input for the chat
text = processor.apply_chat_template(
    messages, tokenize=False, add_generation_prompt=True
)

# Process vision input (image and video)
image_inputs, video_inputs = process_vision_info(messages)

# Prepare model inputs
inputs = processor(
    text=[text],
    images=image_inputs,
    videos=video_inputs,
    padding=True,
    return_tensors="pt",
)
inputs = inputs.to("cuda")

# Generate response
generated_ids = model.generate(**inputs, max_new_tokens=512)

# Extract and decode generated tokens
generated_ids_trimmed = [
    out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
]
output_text = processor.batch_decode(
    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
)

# Print the output
print(output_text)
