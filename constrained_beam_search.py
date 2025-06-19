import torch
from transformers.generation.beam_search import ConstrainedBeamSearchScorer
from transformers import MarianMTModel
from transformers.generation.beam_constraints import DisjunctiveConstraint, PhrasalConstraint
from torch import nn
from transformers import AutoTokenizer
import math
import matplotlib.pyplot as plt


model_name = "Helsinki-NLP/opus-mt-en-uk"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# source_string = 'To continue, tap the button'
# source_string = 'Tap the screen anywhere'
# source_string = 'Tap the screen'
# source_string = 'Tap here to continue'
source_string = 'To continue, tap the button at the top of the screen'
# source_string = 'you are my friend'
processed = tokenizer(source_string, return_tensors="pt", padding=True)
inputs_tensor = processed['input_ids']
force_token_ids = tokenizer(text_target=['торкніться'], add_special_tokens=False).input_ids
# force_words_ids = tokenizer(text_target=['натисніть'], add_special_tokens=False).input_ids
print(force_token_ids)
# 150 ти
# 218 ви

batch_size = 1
num_beams = 5
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

final_constraints = []
if model.generation_config.constraints is not None:
    final_constraints = model.generation_config.constraints
for word_ids in force_token_ids:
    # constraint = DisjunctiveConstraint(word_ids)
    constraint = PhrasalConstraint(word_ids)
    final_constraints.append(constraint)


constrained_beam_scorer = ConstrainedBeamSearchScorer(
    constraints=final_constraints,
    batch_size=batch_size,
    num_beams=num_beams,
    device=inputs_tensor.device,
    length_penalty=model.generation_config.length_penalty,
    do_early_stopping=model.generation_config.early_stopping,
    num_beam_hyps_to_keep=model.generation_config.num_return_sequences,
    max_length=model.generation_config.max_length,
)

input_ids_seq_length = input_ids.shape[-1]
model.generation_config.num_beams = 5
model.generation_config.force_words_ids = [[24818, 9537]]

# from transformers.generation.logits_process import (LogitsProcessorList, NoRepeatNGramLogitsProcessor,
#                                                     NoBadWordsLogitsProcessor, ForcedEOSTokenLogitsProcessor,
#                                                     InfNanRemoveLogitsProcessor)
# logits_processor = LogitsProcessorList()
# logits_processor.append(NoRepeatNGramLogitsProcessor(1))
# logits_processor.append(NoBadWordsLogitsProcessor(model.generation_config.bad_words_ids, model.generation_config.eos_token_id))
# logits_processor.append(ForcedEOSTokenLogitsProcessor(model.generation_config.max_length, model.generation_config.forced_eos_token_id))
# logits_processor.append(InfNanRemoveLogitsProcessor())

# stopping_criteria = model._get_stopping_criteria(
#     generation_config=model.generation_config, stopping_criteria=[]
# )

# pad_token_id = 61586
# eos_token_id = [0]
# output_scores = False
# output_attentions = False
# output_hidden_states = False
# return_dict_in_generate = False

# init attention / hidden states / scores tuples
# scores = None
# decoder_attentions = None
# cross_attentions = None
# decoder_hidden_states = None

# batch_beam_size, cur_len = input_ids.shape

# initialise score of first beam with 0 and the rest with -1e9. This makes sure that only tokens
# of the first beam are considered to avoid sampling the exact same tokens across all beams.
# beam_scores = torch.zeros((batch_size, num_beams), dtype=torch.float, device=input_ids.device)
# beam_scores[:, 1:] = -1e9
# beam_scores = beam_scores.view((batch_size * num_beams,))
# model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

# plot_id = 0
# standard_deviations = []
# while True:
#     model_inputs = model.prepare_inputs_for_generation(input_ids, **model_kwargs)

    # outputs = model(
    #     **model_inputs,
    #     return_dict=True,
    #     output_attentions=output_attentions,
    #     output_hidden_states=output_hidden_states,
    # )

    # next_token_logits = outputs.logits[:, -1, :]
    # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
    # cannot be generated both before and after the `nn.functional.log_softmax` operation.
    # next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)

    # for idx in range(next_token_logits.size()[0]):
    #     phrase = ' '.join([tokenizer.decode(t, skip_special_tokens=True) for t in input_ids[idx]])

        # inputs = next_token_logits[idx]
        # inputs = inputs[inputs != -math.inf]
        # mean = torch.mean(inputs)
        # std = torch.std(inputs)
        # standard_deviations.append(std.detach().numpy())

        # inputs = inputs.detach().numpy()
        # token_score = inputs[24818]



        # token_score = next_token_logits[idx][24818]
        # token_deviation = (token_score - mean)/std
        # if token_deviation > 1:
        # next_token_logits[idx][24818] += 10

        # token_score = next_token_logits[idx][9537]
        # token_deviation = (token_score - mean)/std
        # if token_deviation > 1:
        # next_token_logits[idx][9537] += 10





        # print(mean)
        # print(std)
        # print(token_score)
        # print(token_deviation)
        # print('##################################')

        # next_token_logits[idx][24818] += std * 5
        # next_token_logits[idx][24818] += 10
        # next_token_logits[idx][9537] += std * 5
        # next_token_logits[idx][9537] += 10
        # if plot_id < 511:
        # if plot_id < 50:
            # fig = plt.figure(num=1, clear=True)
            # ax = fig.add_subplot()
            # ax.hist(inputs, ec='lightblue', bins=1000)
            # ax.xaxis.set_visible(False)
            # ax.yaxis.set_visible(False)
            # ax.set_title(phrase)
            # ax.axvline(inputs[24818], color='k', linestyle='dashed', linewidth=1)
            # ax.axvline(inputs[218], color='k', linestyle='dashed', linewidth=1)
            # ax.axvline(inputs[5156], color='red', linestyle='dashed', linewidth=1)
            # ax.axvline(inputs[150], color='red', linestyle='dashed', linewidth=1)
            # plt.savefig(f'plots_big/plot{plot_id}.png')
            # plt.savefig(f'plots_ви/plot{plot_id}.png')
            # plt.savefig(f'plots_adjusted/plot{plot_id}.png')
            # plot_id += 1


    # next_token_scores = nn.functional.log_softmax(
    #     next_token_logits, dim=-1
    # )  # (batch_size * num_beams, vocab_size)

    # next_token_scores_processed = logits_processor(input_ids, next_token_scores)

    # next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

    # scores_for_all_vocab = next_token_scores.clone()

    # reshape for beam search
    # vocab_size = next_token_scores.shape[-1]
    # next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
    # next_token_scores, next_tokens = torch.topk(
    #     next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    # )

    # next_indices = (next_tokens / vocab_size).long()
    # next_tokens = next_tokens % vocab_size

    # stateless
    # check if next token finishes sentence
    # if yes, check if it satisfies the constraint
    # save 5 top tokens that don't finish the sentence
    #
    # beam_outputs = constrained_beam_scorer.process(
    #     input_ids,
    #     next_token_scores,
    #     next_tokens,
    #     next_indices,
    #     scores_for_all_vocab,
    #     pad_token_id=pad_token_id,
    #     eos_token_id=eos_token_id,
    # )
    # beam_scores = beam_outputs["next_beam_scores"]
    # beam_next_tokens = beam_outputs["next_beam_tokens"]
    # beam_idx = beam_outputs["next_beam_indices"]

    # input_ids = torch.cat([input_ids[beam_idx, :], beam_next_tokens.unsqueeze(-1)], dim=-1)
    # model_kwargs = model._update_model_kwargs_for_generation(
    #     outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
    # )
    # if model_kwargs["past_key_values"] is not None:
    #     model_kwargs["past_key_values"] = model._reorder_cache(model_kwargs["past_key_values"], beam_idx)

    # increase cur_len
    # cur_len = cur_len + 1
    # if constrained_beam_scorer.is_done or stopping_criteria(input_ids, scores):
    #     break

# hyps = constrained_beam_scorer._beam_hyps[0]

# for score, sent, _ in hyps.beams:
#     print(score)
#     print(tokenizer.decode(sent, skip_special_tokens=True))

# sequence_outputs = constrained_beam_scorer.finalize(
#     input_ids,
#     beam_scores,
#     next_tokens,
#     next_indices,
#     pad_token_id=pad_token_id,
#     eos_token_id=eos_token_id,
#     max_length=stopping_criteria.max_length,
# )

# print(standard_deviations)
# plt.plot(standard_deviations, 'o')
# plt.scatter(x=list(range(len(standard_deviations))), y=standard_deviations, marker='o')
# plt.xlabel('Prediction number')
# plt.ylabel('Logits standard deviation')
# plt.show()
# standard_deviations = []

# print(sequence_outputs["sequences"])
# print(tokenizer.decode(sequence_outputs["sequences"][0], skip_special_tokens=True))


# unrestricted search with forced word
# add all scores across the sentence, not just the last token

