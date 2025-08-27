import re
from datasets import load_dataset
from datasets.utils.info_utils import VerificationMode

# Load your dataset (replace 'your_dataset_name' with the actual Hugging Face dataset name or path)
ds = load_dataset("NousResearch/Hermes-4-v4-Final-Nonreasoning-Only", verification_mode=VerificationMode.NO_CHECKS, split="train[:20000]")

def process_conversation(row):
    #image_path = row["image"]
    original_conv = row["conversations"]
    
    messages = [
        {
            "role": "system",
            "content": [
                {"type": "text", "text": "You are a helpful assistant."}
            ]
        }
    ]
    
    # Assuming the conversation alternates starting with human, and image is only in the first human message
    for i, turn in enumerate(original_conv):
        role = "user" if turn["from"] == "human" else "assistant"
        value = turn["value"]
        """
        
        if role == "user" and i == 0 and "<image>" in value:
            # Split the value around <image>
            parts = re.split(r'(\n?<image>\n?)', value)
            content = []
            for part in parts:
                if re.match(r'\n?<image>\n?', part):
                    content.append({"type": "image", "path": image_path})
                elif part.strip():
                    content.append({"type": "text", "text": part.strip()})
        else:
        """


        content = [{"type": "text", "text": value.strip(), "path": None}]
        
        messages.append({
            "role": role,
            "content": content
        })
    
    # The format has an outer list with a dict containing "messages"
    #new_conv = [{"messages": messages}]
    
    return {"conversations": messages}

# Apply the transformation and keep only the new "conversations" column
new_ds = ds.map(
    process_conversation,
    remove_columns=ds.column_names
)

# Optionally, push to Hugging Face or save
# new_ds.push_to_hub("new_dataset_name")
# or


print(new_ds[200]['conversations'])
new_ds.save_to_disk("H4_Subset")

#print(new_ds[0])
