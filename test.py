import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
model = RobertaModel.from_pretrained("microsoft/codebert-base-mlm")

from transformers import RobertaConfig, RobertaTokenizer, RobertaForMaskedLM, pipeline

model = RobertaForMaskedLM.from_pretrained("microsoft/codebert-base-mlm")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base-mlm")
CODE = "if (x is not None) <mask> (x>1)"
fill_mask = pipeline('fill-mask', model=model, tokenizer=tokenizer)
outputs = fill_mask(CODE)
print(outputs)