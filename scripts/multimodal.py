from datetime import datetime, timedelta
import torch
import time

from mistral_common.protocol.instruct.request import ChatCompletionRequest
from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
from huggingface_hub import hf_hub_download
from transformers import Mistral3ForConditionalGeneration


def load_system_prompt(repo_id: str, filename: str) -> str:
    file_path = hf_hub_download(repo_id=repo_id, filename=filename)
    with open(file_path, "r") as file:
        system_prompt = file.read()
    today = datetime.today().strftime("%Y-%m-%d")
    yesterday = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")
    model_name = repo_id.split("/")[-1]
    return system_prompt.format(name=model_name, today=today, yesterday=yesterday)


model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"
SYSTEM_PROMPT = load_system_prompt(model_id, "SYSTEM_PROMPT.txt")

tokenizer = MistralTokenizer.from_hf_hub(model_id)

model = Mistral3ForConditionalGeneration.from_pretrained(
    model_id, torch_dtype=torch.bfloat16
)

image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {
        "role": "user",
        "content": [
            {
                "type": "text",
                "text": "What action do you think I should take in this situation? List all the possible actions and explain why you think they are good or bad.",
            },
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
            {"type": "image_url", "image_url": {"url": image_url}},
        ],
    },
]

tokenized = tokenizer.encode_chat_completion(ChatCompletionRequest(messages=messages))

input_ids = torch.tensor([tokenized.tokens])
attention_mask = torch.ones_like(input_ids)

pixel_values = torch.tensor(tokenized.images).to(dtype=torch.bfloat16)
image_sizes = torch.tensor([[pixel_values.shape[-2], pixel_values.shape[-1]]] * len(tokenized.images))


t1 = time.time()

for i in range(10):
    with torch.no_grad():  # For inference efficiency
        image_features = model.get_image_features(
            pixel_values=pixel_values,
            image_sizes=image_sizes
        )

        print(len(image_features))
        print(image_features[0].shape)

        tensor_size_in_bytes = image_features[0].nelement() * image_features[0].element_size()
        print(f"Tensor size: {tensor_size_in_bytes / 1024 / 1024} MB")

t2 = time.time()
print(f"Time taken: {t2 - t1} seconds")
