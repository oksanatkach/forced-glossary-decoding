from transformers import MarianMTModel
from transformers import AutoTokenizer
import torch

def get_next_token(model, encoder_outputs, decoder_input_ids):
    decoder_output = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=decoder_input_ids)
    next_token_logits = decoder_output.logits[:, -1, :].clone().float()
    next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
    max_index = torch.argmax(next_token_scores)
    return max_index.resize(1,1), next_token_scores[max_index].item()


if __name__ == '__main__':
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
    initial_token = torch.tensor([[61586]])
    original_seq = initial_token
    original_score = 0
    original_done = False
    forced_seqs = []
    forced_scores = []
    forced_done = []

    while True:
        if original_done and all(forced_done):
            break

        for ind in range(len(forced_seqs)):
            done = forced_done[ind]
            if not done:
                seq = forced_seqs[ind]
                decoder_output = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=seq)
                next_token_logits = decoder_output.logits[:, -1, :].clone().float()
                next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)
                max_index = torch.argmax(next_token_scores)
                next_token = max_index.resize(1, 1)
                token_score = next_token_scores[:, max_index].item()

                forced_seqs[ind] = torch.cat((seq, next_token), dim=1)
                forced_scores[ind] += token_score

                if max_index.item() == eos_token or len(forced_seqs[ind]) == max_length:
                    forced_done[ind] = True

        if not original_done:
            # get logits
            decoder_output = model.forward(encoder_outputs=encoder_outputs, decoder_input_ids=original_seq)
            next_token_logits = decoder_output.logits[:, -1, :].clone().float()
            next_token_scores = torch.nn.functional.log_softmax(next_token_logits, dim=-1)

            # get forced seq score
            force_tokens_scores = next_token_scores[:, force_token_ids]
            new_forced_seq = torch.cat((original_seq, torch.tensor(force_token_ids)), axis=1)
            new_forced_score = original_score + force_tokens_scores.sum().item()
            forced_seqs.append(new_forced_seq)
            forced_scores.append(new_forced_score)
            forced_done.append(False)

            # process original seq
            max_index = torch.argmax(next_token_scores)
            next_token = max_index.resize(1, 1)
            token_score = next_token_scores[:, max_index].item()
            original_seq = torch.cat((original_seq, next_token), dim=1)
            original_score += token_score
            if max_index.item() == eos_token or len(original_seq) == max_length:
                original_done = True

    print('ORIGINAL')
    print(tokenizer.decode(original_seq[0], skip_special_tokens=True))
    print('###########################################')

    print('FORCED')
    sorted_by_score = sorted(list(enumerate(forced_scores)), key=lambda x: x[1], reverse=True)
    best_ind = sorted_by_score[0][0]
    print(tokenizer.decode(forced_seqs[best_ind][0], skip_special_tokens=True))
