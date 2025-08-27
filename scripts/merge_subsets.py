import re
from datasets import load_dataset, load_from_disk, concatenate_datasets, Features, Value, ClassLabel, List
from datasets.utils.info_utils import VerificationMode


new_features = Features({'conversations': List({'content': List({'path': Value('string'), 'text': Value('string'), 'type': Value('string')}), 'role': Value('string')})})


ds1 = load_from_disk("ChartQA_Subset")
ds2 = load_from_disk("H4_Subset").cast(new_features)


ds3 = concatenate_datasets([ds1, ds2])

print(ds3[0])

ds3.save_to_disk("CombinedDataset")



