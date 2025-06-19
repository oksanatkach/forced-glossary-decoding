import torch
from transformers.generation.beam_search import ConstrainedBeamSearchScorer
from transformers import MarianMTModel
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from torch import nn
from transformers import AutoTokenizer
from testing_2 import align_hyp_and_requirement, logits_to_scores, is_sublist
import math

model_name = "Helsinki-NLP/opus-mt-en-uk"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

source_string = 'To continue, tap the button'
# source_string = 'Tap the screen anywhere'
processed = tokenizer(source_string, return_tensors="pt", padding=True)
inputs_tensor = processed['input_ids']
force_words_ids = tokenizer(text_target=['торкніться'], add_special_tokens=False).input_ids
# force_words_ids = tokenizer(text_target=['натисніть'], add_special_tokens=False).input_ids

batch_size = 1
num_beams = 20
model_input_name = 'input_ids'
model_kwargs = {
    'output_attentions': False,
    'output_hidden_states': False,
    'use_cache': True,
    'attention_mask': torch.ones(1, inputs_tensor.size()[1]),
}
model_kwargs = model._prepare_encoder_decoder_kwargs_for_generation(
    inputs_tensor, model_kwargs, model_input_name
)
input_ids = torch.tensor([[61586]])
input_ids, model_kwargs = model._expand_inputs_for_generation(
    input_ids=input_ids,
    expand_size=num_beams,
    is_encoder_decoder=model.config.is_encoder_decoder,
    **model_kwargs,
)

# final_constraints = []
# if model.generation_config.constraints is not None:
#     final_constraints = model.generation_config.constraints
# for word_ids in force_words_ids:
    # constraint = DisjunctiveConstraint(word_ids)
    # constraint = PhrasalConstraint(word_ids)
    # final_constraints.append(constraint)


# constrained_beam_scorer = ConstrainedBeamSearchScorer(
#     constraints=final_constraints,
#     batch_size=batch_size,
#     num_beams=num_beams,
#     device=inputs_tensor.device,
#     length_penalty=model.generation_config.length_penalty,
#     do_early_stopping=model.generation_config.early_stopping,
#     num_beam_hyps_to_keep=model.generation_config.num_return_sequences,
#     max_length=model.generation_config.max_length,
# )

# input_ids_seq_length = input_ids.shape[-1]
# model.generation_config.num_beams = 5
# model.generation_config.force_words_ids = [[24818, 9537]]

from transformers.generation.logits_process import (LogitsProcessorList, NoRepeatNGramLogitsProcessor,
                                                    NoBadWordsLogitsProcessor, ForcedEOSTokenLogitsProcessor,
                                                    InfNanRemoveLogitsProcessor)
logits_processor = LogitsProcessorList()
logits_processor.append(NoRepeatNGramLogitsProcessor(1))
logits_processor.append(NoBadWordsLogitsProcessor(model.generation_config.bad_words_ids, model.generation_config.eos_token_id))
logits_processor.append(ForcedEOSTokenLogitsProcessor(model.generation_config.max_length, model.generation_config.forced_eos_token_id))
logits_processor.append(InfNanRemoveLogitsProcessor())

# stopping_criteria = model._get_stopping_criteria(
#     generation_config=model.generation_config, stopping_criteria=[]
# )

pad_token_id = 61586
eos_token_id = [0]
output_scores = False
output_attentions = False
output_hidden_states = False
return_dict_in_generate = False

# init attention / hidden states / scores tuples
scores = None
decoder_attentions = None
cross_attentions = None
decoder_hidden_states = None

batch_beam_size, cur_len = input_ids.shape

# initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
# of the first beam are considered to avoid sampling the exact same tokens across all beams.
beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
# -1e9 is minus one billion
beam_scores[:, 1:] = -1e9
beam_scores = beam_scores.view((batch_size * num_beams,))

hyp_scores = []
hyp_tokens = []
finished_hyps = []
while True:
    model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    outputs = model(
        **model_inputs,
        return_dict=True,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
    )

    next_tokens, next_token_scores, next_indices, scores_for_all_vocab = logits_to_scores(outputs, model,
                                                                    logits_processor, input_ids, beam_scores,
                                                                    batch_size, num_beams, cur_len)

    # run constrained beam search in parallel

    # advanced forced token:
    # run 1: force first required token to start the sequence, memorize the score

    # run 1
    # if this is the first predicted token in the loop:
    if cur_len == 1:
        # force the first token of a the requirement as the first token of prediction
        if force_words_ids[0][0] not in next_tokens:
            next_tokens = torch.cat((torch.tensor([force_words_ids[0][0]]), next_tokens[0]))
            next_token_scores = torch.cat((torch.tensor([scores_for_all_vocab[0][force_words_ids[0][0]]]),
                                           next_token_scores[0]))
            next_indices = torch.cat((torch.tensor([0]), next_indices[0]))
        else:
            next_tokens = next_tokens[0]
            next_token_scores = next_token_scores[0]
            next_indices = next_indices[0]

        logical_selector = torch.tensor(range(num_beams))
        next_tokens = torch.index_select(next_tokens, 0, logical_selector)
        next_token_scores = torch.index_select(next_token_scores, 0, logical_selector)
        next_indices = torch.index_select(next_indices, 0, logical_selector)

        # move?
        beam_scores = next_token_scores
        beam_next_tokens = next_tokens
        beam_idx = next_indices



    # run 2 and onward
    else:
        # end of sentence
        eos_tokens = [token in eos_token_id for token in next_tokens[0]]
        eos_tokens = torch.tensor(eos_tokens, dtype=torch.bool)
        if any(eos_tokens):
            eos_indices = (eos_tokens == True).nonzero()
            eos_indices = torch.flatten(eos_indices)
            eos_scores = torch.index_select(next_token_scores, 1, eos_indices)
            eos_beam_indices = torch.index_select(next_indices, 1, eos_indices)

            for lst_ind, beam_ind in enumerate(eos_beam_indices[0]):
                if is_sublist(input_ids[beam_ind].tolist(), force_words_ids[0]):
                    finished_hyps.append((eos_scores[0][lst_ind], input_ids[beam_ind].tolist()+[0]))

                    print(eos_scores[0][lst_ind].item())
                    tokens = input_ids[beam_ind].tolist()+[0]
                    print(tokenizer.decode(tokens, skip_special_tokens=True))

            new_next_tokens = []
            new_next_token_scores = []
            new_next_indices = []

            for ind, token in enumerate(next_tokens[0]):
                if ind not in eos_indices:
                    new_next_tokens.append(token)
                    new_next_token_scores.append(next_token_scores[0][ind])
                    new_next_indices.append(next_indices[0][ind])

            next_tokens = torch.tensor([new_next_tokens])
            next_token_scores = torch.tensor([new_next_token_scores])
            next_indices = torch.tensor([new_next_indices])

            # get score by index
            # get beam by index
            # retrieve appropriate beam
            # save final hyp and score




        # run 2:
        # force next token to the previous forced hypothesis
        # force the first token to each top k prediction
        # sum up scores of each token in the sequence
        # get top k sequences each run?

        hyps_forced_tokens = []
        hyps_forced_scores = []
        hyps_forced_indices = []

        hyps_predicted_tokens = []
        hyps_predicted_scores = []
        hyps_predicted_indices = []

        for index, hyp in enumerate(input_ids):
            requirement_satisfied, requirement_finished, next_token = align_hyp_and_requirement(hyp, force_words_ids[0])

            # if requirement is started but not finished, force the next token
            if requirement_satisfied and not requirement_finished:
                hyps_forced_tokens.append(force_words_ids[0][next_token])
                hyps_forced_scores.append(beam_scores[index] + scores_for_all_vocab[0][force_words_ids[0][next_token]])
                hyps_forced_indices.append(index)

            # if the forced phrase is already in the hypothesis, just memorize it
            elif requirement_finished:
                # get all predicted tokens for this beam
                next_indices_this_beam = (next_indices == index).nonzero()
                next_indices_this_beam = torch.flatten(next_indices_this_beam)
                next_tokens_this_beam = torch.index_select(next_tokens, 1, next_indices_this_beam)
                next_scores_this_beam = torch.index_select(next_token_scores, 1, next_indices_this_beam)

                hyps_forced_tokens += next_tokens_this_beam.tolist()[0]
                hyps_forced_scores += next_scores_this_beam.tolist()[0]
                hyps_forced_indices += [index] * len(next_indices_this_beam)

            # if the hypothesis doesn't contain the requirement yet, memorize the hyp as is, and another version
            # with the first token of the requirement forced
            elif not requirement_satisfied:
                next_indices_this_beam = (next_indices[0] == index).nonzero()
                next_indices_this_beam = torch.flatten(next_indices_this_beam)
                next_tokens_this_beam = torch.index_select(next_tokens, 1, next_indices_this_beam)
                next_scores_this_beam = torch.index_select(next_token_scores, 1, next_indices_this_beam)

                hyps_predicted_tokens += next_tokens_this_beam.tolist()[0]
                hyps_predicted_scores += next_scores_this_beam.tolist()[0]
                hyps_predicted_indices += [index] * len(next_indices_this_beam)

                hyps_forced_tokens.append(force_words_ids[0][0])
                hyps_forced_scores.append(beam_scores[index] + scores_for_all_vocab[0][force_words_ids[0][0]])
                hyps_forced_indices.append(index)

        # narrow down to 20
        hyps_forced_scores_tpl = enumerate(hyps_forced_scores)
        hyps_forced_scores_filtered = filter(lambda tpl: tpl[1] != -math.inf, hyps_forced_scores_tpl)
        hyps_forced_scores_sorted = sorted(hyps_forced_scores_filtered, key=lambda tpl: tpl[1], reverse=True)
        forced_cutoff = len(hyps_forced_scores_sorted) if len(hyps_forced_scores_sorted) < 10 else 10
        hyps_forced_to_push = hyps_forced_scores_sorted[:forced_cutoff]
        if hyps_forced_to_push:
            unzip_forced_scores = list(zip(*hyps_forced_to_push))
            forced_item_indices_to_keep = list(unzip_forced_scores[0])
            hyps_forced_scores_to_keep = list(unzip_forced_scores[1])
            hyps_forced_tokens_to_keep = [hyps_forced_tokens[ind] for ind in forced_item_indices_to_keep]
            hyps_forced_indices_to_keep = [hyps_forced_indices[ind] for ind in forced_item_indices_to_keep]
        else:
            hyps_forced_scores_to_keep = []
            hyps_forced_tokens_to_keep = []
            hyps_forced_indices_to_keep = []

        hyps_predicted_scores_tpl = enumerate(hyps_predicted_scores)
        hyps_predicted_scores_sorted = sorted(hyps_predicted_scores_tpl, key=lambda tpl: tpl[1], reverse=True)
        predicted_cutoff = 20-forced_cutoff if hyps_forced_to_push else 20
        hyps_predicted_to_push = hyps_predicted_scores_sorted[:predicted_cutoff]
        unzip_predicted_scores = list(zip(*hyps_predicted_to_push))
        predicted_item_indices_to_keep = list(unzip_predicted_scores[0])
        hyps_predicted_scores_to_keep = list(unzip_predicted_scores[1])
        hyps_predicted_tokens_to_keep = [hyps_predicted_tokens[ind] for ind in predicted_item_indices_to_keep]
        hyps_predicted_indices_to_keep = [hyps_predicted_indices[ind] for ind in predicted_item_indices_to_keep]


        beam_scores = torch.tensor(hyps_forced_scores_to_keep+hyps_predicted_scores_to_keep)
        beam_next_tokens = torch.tensor(hyps_forced_tokens_to_keep+hyps_predicted_tokens_to_keep)
        beam_idx = torch.tensor(hyps_forced_indices_to_keep+hyps_predicted_indices_to_keep)

        if len(beam_scores) < 20:
            num_to_add = 20 - len(beam_scores)
            if hyps_forced_scores and not list(hyps_forced_scores_filtered):
                beam_scores = torch.cat((beam_scores, torch.tensor(hyps_forced_scores[:num_to_add])))
                beam_next_tokens = torch.cat((beam_next_tokens, torch.tensor(hyps_forced_tokens[:num_to_add])))
                beam_idx = torch.cat((beam_idx, torch.tensor(hyps_forced_indices[:num_to_add])))

        if len(beam_scores) < 20:
            print(True)

    # print(new_hyp_tokens)

    # implement run 2+ token prediction to test this
    # narrow all variants down to 5? 10?

    input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    model_kwargs = model._update_model_kwargs_for_generation(
        outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    )
    if model_kwargs["past_key_values"] is not None:
        model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

    # increase cur_len
    cur_len = cur_len + 1
    if cur_len == 513:
        break

print(finished_hyps)
