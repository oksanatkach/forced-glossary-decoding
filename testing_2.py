from torch import nn
import torch


def align_hyp_and_requirement(hyp, requirement):
    '''
    find which part of requirement is already satisfied in the hypothesis
    :param hyp: sequence of tokens from the beam search
    :param requirement: sequence of required tokens
    :return: next token to add
    '''
    # find all occurances of the last token in hypothesis in the required phrase
    indices = [i for i, x in enumerate(requirement) if x == hyp[-1]]
    winner_index = None

    if indices:
        for _index in indices:
            hyp_stack = hyp.tolist()
            for force_sequence_index in range(_index, -1, -1):
                next_force_word = requirement[force_sequence_index]
                last_hyp_word = hyp_stack.pop()
                if next_force_word != last_hyp_word:
                    break
                if force_sequence_index == 0:
                    winner_index = _index

    requirement_satisfied = winner_index is not None
    requirement_finished = winner_index+1 == len(requirement) if requirement_satisfied else False
    next_token = None if (requirement_satisfied and requirement_finished)\
                      or (not requirement_satisfied)\
                      else winner_index+1

    return requirement_satisfied, requirement_finished, next_token


def is_sublist(lst, sublst):
    completed = False
    ind = 0
    for el_lst in lst:
        if ind == len(sublst):
            completed = True
        else:
            el_sublst = sublst[ind]
            if el_lst == el_sublst:
                ind += 1
            else:
                ind = 0

    return completed


def logits_to_scores(outputs, model, logits_processor, input_ids, beam_scores, batch_size, num_beams, cur_len):
    next_token_logits = outputs.logits[:, -1, :]
    # hack: adjust tokens for Marian. For Marian we have to make sure that the `pad_token_id`
    # cannot be generated both before and after the `nn.functional.log_softmax` operation.
    next_token_logits = model.adjust_logits_during_generation(next_token_logits, cur_len=cur_len)

    for idx in range(next_token_logits.size()[0]):
        next_token_logits[idx][24818] += 10
        next_token_logits[idx][9537] += 10

    next_token_scores = nn.functional.log_softmax(
        next_token_logits, dim=-1
    )  # (batch_size * num_beams, vocab_size)

    next_token_scores_processed = logits_processor(input_ids, next_token_scores)

    # next_token_scores = next_token_scores_processed
    next_token_scores = next_token_scores_processed + beam_scores[:, None].expand_as(next_token_scores)

    scores_for_all_vocab = next_token_scores.clone()

    # reshape for beam search
    vocab_size = next_token_scores.shape[-1]
    next_token_scores = next_token_scores.view(batch_size, num_beams * vocab_size)

    # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
    next_token_scores, next_tokens = torch.topk(
        next_token_scores, 2 * num_beams, dim=1, largest=True, sorted=True
    )

    # earlier all vocab per beam is merged
    # this figures out which beam the token id belongs to by dividing by vocab length
    next_indices = (next_tokens / vocab_size).long()
    next_tokens = next_tokens % vocab_size

    return next_tokens, next_token_scores, next_indices, scores_for_all_vocab
