# import logging
# from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
# from qwen_vl_utils import process_vision_info
# import torch

# # Setup logger
# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
#     handlers=[
#         logging.FileHandler("qwen_vl_inference.log"),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger(__name__)

# # Load model
# logger.info("Loading model...")
# model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
#     "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
# )
# logger.info("Model loaded successfully.")

# # Load processor
# logger.info("Loading processor...")
# processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
# logger.info("Processor loaded successfully.")

# # Message input
# messages = [
#     {
#         "role": "user",
#         "content": [
#             {
#                 "type": "image",
#                 "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
#             },
#             {"type": "text", "text": "Describe this image."},
#         ],
#     }
# ]

# logger.info("Applying chat template...")
# text = processor.apply_chat_template(
#     messages, tokenize=False, add_generation_prompt=True
# )

# logger.info("Processing vision input...")
# image_inputs, video_inputs = process_vision_info(messages)

# logger.info("Tokenizing inputs...")
# inputs = processor(
#     text=[text],
#     images=image_inputs,
#     videos=video_inputs,
#     padding=True,
#     return_tensors="pt",
# )
# # inputs = inputs.to("cuda")

# logger.info("Running inference...")
# generated_ids = model.generate(**inputs, max_new_tokens=128)

# logger.info("Postprocessing output...")
# generated_ids_trimmed = [
#     out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
# ]
# output_text = processor.batch_decode(
#     generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
# )

# logger.info(f"Output: {output_text}")





import logging
import torch
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from qwen_vl_utils import process_vision_info

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("qwen_vl_process.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

try:
    # Load model
    logger.info("Loading Qwen2.5-VL model...")
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2.5-VL-3B-Instruct", torch_dtype="auto", device_map="auto"
    )
    logger.info("Model loaded successfully.")

    # Load processor
    logger.info("Loading processor...")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    logger.info("Processor loaded successfully.")

    # Input message
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
                },
                {"type": "text", "text": "Describe this image."},
            ],
        }
    ]

    # Preprocessing
    logger.info("Applying chat template to the message...")
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    logger.info("Processing vision info from the message...")
    image_inputs, video_inputs = process_vision_info(messages)

    logger.info("Preparing model inputs...")
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    # inputs = inputs.to("cuda")

    # Inference
    logger.info("Generating output from the model...")
    generated_ids = model.generate(**inputs, max_new_tokens=128)

    logger.info("Postprocessing generated output...")
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )

    logger.info(f"Final Output: {output_text}")
    print(output_text)

except Exception as e:
    logger.exception(f"An error occurred during processing: {e}")










































