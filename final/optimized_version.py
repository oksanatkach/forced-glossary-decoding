from transformers import MarianMTModel
from transformers import AutoTokenizer
import torch

model_name = "Helsinki-NLP/opus-mt-en-uk"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
source_string = 'To continue, tap the button at the top of the screen'
processed = tokenizer(source_string, return_tensors="pt", padding=True)
inputs_tensor = processed['input_ids']
force_token_ids = tokenizer(text_target=['торкніться'], add_special_tokens=False).input_ids
encoder_outputs = model.model.encoder(input_ids=inputs_tensor)

eos_token = 0
max_length = 512
seq = torch.tensor([[61586]])
done = False
ind = 0
all_scores = []
while not done or ind < max_length:
    decoder_output = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=seq)
    next_token_logits = decoder_output.logits[:, -1, :].clone().float()
    next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    ind += 1
