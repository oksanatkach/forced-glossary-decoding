from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.tf_logits_process import TFLogitsProcessorList
from transformers.tf_utils import shape_list
from transformers import TFMarianMTModel, AutoTokenizer
import tensorflow as tf
import inspect


strings = [
    ('Tap the screen anywhere', 'Торкніться екрана будь-де'),
    ('Tap here to continue', 'Торкніться тут, щоб продовжити'),
    ('tap', 'Торкнутись'),
    ('he tapped out', 'він здався'),
    ('To continue, tap the button at the top of the screen', 'торкніться кнопки зверху екрана')
]

logits_processor = TFLogitsProcessorList()
logits_warper = TFLogitsProcessorList()

model_name = "Helsinki-NLP/opus-mt-en-uk"
model = TFMarianMTModel.from_pretrained(model_name)

generation_config = GenerationConfig(**{
  "_from_model_config": True,
  "bad_words_ids": [
    [
      61586
    ]
  ],
  "bos_token_id": 0,
  "decoder_start_token_id": 61586,
  "eos_token_id": 0,
  "forced_eos_token_id": 0,
  "max_length": 512,
  "num_beams": 4,
  "pad_token_id": 61586,
  "transformers_version": "4.29.2"
})

tokenizer = AutoTokenizer.from_pretrained(model_name)
# input_tensor = tokenizer(strings[0][0], return_tensors="tf", padding=True)['input_ids']
# input_tensor = tokenizer('please tap here', return_tensors="tf", padding=True)['input_ids']
input_tensor = tokenizer(strings[4][0], return_tensors="tf", padding=True)['input_ids']
# print(input_tensor)

attention_mask = tf.ones((1, tuple(input_tensor.shape)[1]))

model_kwargs = {
  'attention_mask': attention_mask,
  'output_attentions': False,
  'output_hidden_states': False,
  'use_cache': True
}
model_kwargs["attention_mask"] = tf.cast(model_kwargs["attention_mask"], tf.int32)


input_ids = tf.constant(
[[[61586],
  [61586],
  [61586],
  [61586]]], dtype=tf.int32)

input_ids_seq_length = 1
has_default_max_length = True
model_input_name = 'input_ids'
batch_size = 1
accepts_attention_mask = True
requires_attention_mask = True


encoder = model.get_encoder()

# 2. prepare encoder args and encoder kwargs from model kwargs
irrelevant_prefix = ["decoder_", "cross_attn", "use_cache"]
encoder_kwargs = {
  argument: value
  for argument, value in model_kwargs.items()
  if not any(argument.startswith(p) for p in irrelevant_prefix)
}
encoder_signature = set(inspect.signature(encoder.call).parameters)
encoder_accepts_wildcard = "kwargs" in encoder_signature or "model_kwargs" in encoder_signature
if not encoder_accepts_wildcard:
  encoder_kwargs = {
    argument: value for argument, value in encoder_kwargs.items() if argument in encoder_signature
  }

# 3. vision models don't use `attention_mask`.
encoder_kwargs["return_dict"] = True
encoder_kwargs[model_input_name] = input_tensor
if model_input_name != model.main_input_name:  # in Keras, the first input must always be passed
  encoder_kwargs[model.main_input_name] = None
# print(encoder_kwargs)
encoder_outputs = encoder(**encoder_kwargs)

####################################################### encode inputs
model_kwargs["encoder_outputs"] = encoder_outputs

is_contrastive_search_gen_mode = False
is_greedy_gen_mode = False
is_beam_gen_mode = True
is_sample_gen_mode = False
is_beam_sample_gen_mode = False

model_kwargs = {
  'attention_mask': attention_mask,
  'output_attentions': False,
  'output_hidden_states': False,
  'use_cache': True,
  'encoder_outputs': model_kwargs['encoder_outputs']
}
expand_size = 4


def _expand_tensor(tensor: tf.Tensor):
  shape = shape_list(tensor)
  return tf.broadcast_to(tensor[:, None], (shape[0], expand_size) + tuple(shape[1:]))


def _expand_dict_for_generation(dict_to_expand):
  for key in dict_to_expand:
    if dict_to_expand[key] is not None and isinstance(dict_to_expand[key], tf.Tensor):
      dict_to_expand[key] = _expand_tensor(dict_to_expand[key])
  return dict_to_expand


model_kwargs = _expand_dict_for_generation(model_kwargs)
model_kwargs["encoder_outputs"] = _expand_dict_for_generation(model_kwargs["encoder_outputs"])

max_length = 512
pad_token_id = 61586
eos_token_id = [0]
num_return_sequences = 1
output_attentions = False
output_hidden_states = False
output_scores = False
return_dict_in_generate = False
early_stopping = False
length_penalty = 1.0
use_cache = True
use_xla = False
cache_batch_axis = 0
needs_full_input = False
all_scores = None
decoder_attentions = None
cross_attentions = None
decoder_hidden_states = None
batch_size = 1
num_beams = 4
cur_len = 1
input_ids_padding = tf.ones((batch_size, num_beams, max_length - cur_len), dtype=tf.int32) * (
        pad_token_id or 0
)
running_sequences = tf.concat([input_ids, input_ids_padding], axis=-1)
sequences = tf.ones((batch_size, num_beams, max_length), dtype=tf.int32) * (pad_token_id or 0)
is_sent_finished = tf.zeros((batch_size, num_beams), dtype=tf.bool)
running_scores = tf.tile(
  tf.expand_dims(tf.convert_to_tensor([0.0] + [-1.0e9] * (num_beams - 1)), axis=0), [batch_size, 1]
)
scores = tf.ones((batch_size, num_beams)) * -1.0e9
running_beam_indices = tf.ones((batch_size, num_beams, max_length), dtype=tf.int32) * -1
beam_indices = tf.ones((batch_size, num_beams, max_length), dtype=tf.int32) * -1


def flatten_beam_dim(tensor, batch_axis=0):
  """Flattens the first two dimensions of a non-scalar array."""
  shape = shape_list(tensor)
  return tf.reshape(
    tensor,
    shape[:batch_axis] + [shape[batch_axis] * shape[batch_axis + 1]] + shape[batch_axis + 2:],
  )


# flatten beam dim
if "encoder_outputs" in model_kwargs:
  model_kwargs["encoder_outputs"]["last_hidden_state"] = flatten_beam_dim(
    model_kwargs["encoder_outputs"]["last_hidden_state"]
  )
if "attention_mask" in model_kwargs:
  model_kwargs["attention_mask"] = flatten_beam_dim(model_kwargs["attention_mask"])


def prepare_inputs_for_generation(
        decoder_input_ids,
        past_key_values=None,
        attention_mask=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        cross_attn_head_mask=None,
        use_cache=None,
        encoder_outputs=None,
        **kwargs,
):
  # cut decoder_input_ids if past_key_values is used
  if past_key_values is not None:
    decoder_input_ids = decoder_input_ids[:, -1:]

  if decoder_attention_mask is not None:  # xla
    decoder_position_ids = tf.math.cumsum(decoder_attention_mask, axis=-1, exclusive=True)[:, -1:]
  elif past_key_values is not None:  # no xla + past_key_values
    decoder_position_ids = past_key_values[0][0].shape[2]
  else:  # no xla + no past_key_values
    decoder_position_ids = tf.range(decoder_input_ids.shape[1])

  return {
    "input_ids": None,  # encoder_outputs is defined. input_ids not needed
    "encoder_outputs": encoder_outputs,
    "past_key_values": past_key_values,
    "decoder_input_ids": decoder_input_ids,
    "attention_mask": attention_mask,
    "decoder_attention_mask": decoder_attention_mask,
    "decoder_position_ids": decoder_position_ids,
    "head_mask": head_mask,
    "decoder_head_mask": decoder_head_mask,
    "cross_attn_head_mask": cross_attn_head_mask,
    "use_cache": use_cache  # change this to avoid caching (presumably for debugging)
  }


def unflatten_beam_dim(tensor, num_beams, batch_axis=0):
  """Unflattens the first, flat batch*beam dimension of a non-scalar array."""
  shape = shape_list(tensor)
  return tf.reshape(tensor, shape[:batch_axis] + [-1, num_beams] + shape[batch_axis + 1:])


def _gather_beams(nested, beam_indices, batch_axis=0):
  """Gathers the beam slices indexed by beam_indices into new beam array."""

  def gather_fn(tensor):
    if batch_axis > 0:
      # pushes all dimentions before the batch to the end, so we get (batch, beam_id, ...)
      perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
      tensor = tf.transpose(tensor, perm=perm)

    gathered_tensor = tf.gather(params=tensor, indices=beam_indices, axis=1, batch_dims=1)
    if batch_axis > 0:
      # transposes back to the original dimensions
      perm = tf.concat((tf.range(tf.rank(tensor))[batch_axis:], tf.range(batch_axis)), axis=0)
      perm = tf.math.invert_permutation(perm)
      gathered_tensor = tf.transpose(gathered_tensor, perm=perm)

    return gathered_tensor

  return tf.nest.map_structure(gather_fn, nested)


def _extract_past_from_model_output(outputs):
  past_key_values = None
  if "past_key_values" in outputs:
    past_key_values = outputs.past_key_values
  elif "mems" in outputs:
    past_key_values = outputs.mems
  elif "past_buckets_states" in outputs:
    past_key_values = outputs.past_buckets_states
  return past_key_values


def _update_model_kwargs_for_generation(outputs, model_kwargs, is_encoder_decoder: bool = False):
  # update past_key_values
  model_kwargs["past_key_values"] = _extract_past_from_model_output(outputs)

  # update attention mask
  if not is_encoder_decoder:
    if "attention_mask" in model_kwargs:
      attention_mask = model_kwargs["attention_mask"]
      model_kwargs["attention_mask"] = tf.concat(
        [attention_mask, tf.ones((shape_list(attention_mask)[0], 1), dtype=tf.int32)], axis=-1
      )

  return model_kwargs


def beam_search_body_fn(
        cur_len,
        running_sequences,
        running_scores,
        running_beam_indices,
        sequences,
        scores,
        beam_indices,
        is_sent_finished,
        model_kwargs,
):
  """
  Beam Search iterative update function -- each iteration adds a new token and updates the best sequences
  seen so far
  """
  # 1. Forward current tokens
  if model_kwargs.get("past_key_values") is None or needs_full_input:
    input_ids = running_sequences[:, :, :cur_len]
  else:
    input_ids = tf.expand_dims(running_sequences[:, :, cur_len - 1], -1)

  model_inputs = prepare_inputs_for_generation(
    flatten_beam_dim(input_ids), **model_kwargs
  )
  model_outputs = model(
    **model_inputs,
    return_dict=True,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
    my_tuning_process=True
  )
  # print(model_outputs)








  logits = unflatten_beam_dim(model_outputs.logits[:, -1], num_beams)

  # 2. Compute log probs
  # get log probabilities from logits, process logits with processors (*e.g.* min_length, ...), and
  # add new logprobs to existing running logprobs scores.
  log_probs = tf.nn.log_softmax(logits)
  log_probs = logits_processor(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), cur_len)
  log_probs = unflatten_beam_dim(log_probs, num_beams)
  log_probs = log_probs + tf.expand_dims(running_scores, axis=2)
  # Note: logits warpers are intentionally applied after adding running beam scores. On some logits
  # warpers (like top_p) this is indiferent, but on others (like temperature) it is not. For reference,
  # see https://github.com/huggingface/transformers/pull/5420#discussion_r449779867
  log_probs = logits_warper(flatten_beam_dim(running_sequences), flatten_beam_dim(log_probs), cur_len)
  log_probs = unflatten_beam_dim(log_probs, num_beams)
  vocab_size = log_probs.shape[2]
  log_probs = tf.reshape(log_probs, (batch_size, num_beams * vocab_size))









  # 3. Retrieve top-K
  # Each item in batch has num_beams * vocab_size candidate sequences. For each item, get the top 2*k
  # candidates with the highest log-probabilities. We gather the top 2*K beams here so that even if the
  # best K sequences reach EOS simultaneously, we have another K sequences remaining to continue the live
  # beam search.
  # Gather the top 2*K scores from _all_ beams.
  # Gather 2*k top beams.
  # Recover the beam index by floor division.
  # Recover token id by modulo division and expand Id array for broadcasting.
  # Update sequences for the 2*K top-k new sequences.

  beams_to_keep = 2 * num_beams
  topk_log_probs, topk_indices = tf.math.top_k(log_probs, k=beams_to_keep)
  topk_current_beam_indices = topk_indices // vocab_size
  topk_running_beam_indices = _gather_beams(running_beam_indices, topk_current_beam_indices)
  topk_running_sequences = _gather_beams(running_sequences, topk_current_beam_indices)
  topk_ids = topk_indices % vocab_size

  # writes the new token
  indices_batch = tf.repeat(tf.range(batch_size), [beams_to_keep])
  indices_beam = tf.tile(tf.range(beams_to_keep), [batch_size])
  update_indices = tf.stack(
    [indices_batch, indices_beam, tf.broadcast_to(cur_len, [batch_size * beams_to_keep])], axis=-1
  )
  topk_sequences = tf.tensor_scatter_nd_update(
    tensor=topk_running_sequences,
    indices=update_indices,
    updates=tf.reshape(topk_ids, [batch_size * beams_to_keep]),
  )

  # we want to store the beam indices with batch information -> real beam index = beam index % num beams
  batch_modified_indices = topk_current_beam_indices + tf.broadcast_to(
    tf.expand_dims(tf.range(batch_size) * num_beams, axis=1), topk_current_beam_indices.shape
  )
  topk_beam_indices = tf.tensor_scatter_nd_update(
    tensor=topk_running_beam_indices,
    indices=update_indices,
    updates=tf.reshape(batch_modified_indices, [batch_size * beams_to_keep]),
  )

  # 4. Check which sequences have ended
  # Update current sequences: Did the top `num_beams` sequences reach an end marker?
  # To prevent these just finished sequences from being added to the current sequences
  # set of active beam search sequences, set their log probs to a very large negative value.
  if eos_token_id is None:
    eos_in_next_token = tf.zeros(topk_sequences[:, :, cur_len].shape, dtype=tf.bool)
  else:
    eos_in_next_token = tf.math.reduce_any(
      tf.equal(
        tf.broadcast_to(
          topk_sequences[:, :, cur_len], [len(eos_token_id)] + topk_sequences[:, :, cur_len].shape
        ),
        tf.expand_dims(tf.expand_dims(eos_token_id, -1), -1),
      ),
      axis=0,
    )
  did_topk_just_finished = eos_in_next_token & tf.broadcast_to(
    tf.concat((tf.ones((num_beams), dtype=tf.bool), tf.zeros((num_beams), dtype=tf.bool)), axis=0),
    shape_list(eos_in_next_token),
  )

  # non-top `num_beams` eos tokens can't be used to finish a beam, but the others can't be used in the next
  # running sentences either
  running_topk_log_probs = topk_log_probs + tf.cast(eos_in_next_token, tf.float32) * -1.0e9

  # 5. Get running sequences scores for next
  # Determine the top k beam indices (from top 2*k beams) from log probs and gather top k beams
  # (from top 2*k beams).
  next_topk_indices = tf.math.top_k(running_topk_log_probs, k=num_beams)[1]
  next_running_sequences, next_running_scores, next_running_beam_indices = _gather_beams(
    [topk_sequences, running_topk_log_probs, topk_beam_indices], next_topk_indices
  )

  # 6. Process topk logits
  # Further process log probs:
  # - add length penalty
  # - make sure no scores can be added anymore if beam is full
  # - make sure still running sequences cannot be chosen as finalized beam
  topk_log_probs = topk_log_probs / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
  beams_in_batch_are_full = tf.broadcast_to(
    tf.math.reduce_all(is_sent_finished, axis=-1, keepdims=True), shape_list(did_topk_just_finished)
  ) & (early_stopping is True)
  add_penalty = ~did_topk_just_finished | beams_in_batch_are_full
  topk_log_probs += tf.cast(add_penalty, tf.float32) * -1.0e9

  # 7. Get scores, sequences, is sentence finished for next.
  # Combine sequences, scores, and flags along the beam dimension and compare new finished sequence scores
  # to existing finished scores and select the best from the new set of beams
  merged_sequences = tf.concat([sequences, topk_sequences], axis=1)
  merged_scores = tf.concat([scores, topk_log_probs], axis=1)
  merged_beams = tf.concat([beam_indices, topk_beam_indices], axis=1)
  merged_is_sent_finished = tf.concat([is_sent_finished, did_topk_just_finished], axis=1)
  topk_merged_indices = tf.math.top_k(merged_scores, k=num_beams)[1]
  next_sequences, next_scores, next_beam_indices, next_is_sent_finished = _gather_beams(
    [merged_sequences, merged_scores, merged_beams, merged_is_sent_finished], topk_merged_indices
  )

  # 8. Prepare data for the next iteration
  # Determine the top k beam indices from the original set of all beams. With these, gather the top k
  # beam-associated caches.
  cur_len = cur_len + 1
  if "past_key_values" in model_outputs:
    cache = tf.nest.map_structure(
      lambda tensor: unflatten_beam_dim(tensor, num_beams, batch_axis=cache_batch_axis),
      model_outputs.past_key_values,
    )
    next_running_indices = _gather_beams(topk_current_beam_indices, next_topk_indices)
    next_cache = _gather_beams(cache, next_running_indices, batch_axis=cache_batch_axis)
    model_outputs["past_key_values"] = tf.nest.map_structure(
      lambda tensor: flatten_beam_dim(tensor, batch_axis=cache_batch_axis), next_cache
    )

  next_model_kwargs = _update_model_kwargs_for_generation(
    model_outputs, model_kwargs, is_encoder_decoder=model.config.is_encoder_decoder
  )

  # if we don't cache past_key_values key values we need the whole input
  if model_kwargs.get("past_key_values", None) is None:
    # let's throw out `past_key_values` since we don't want `None` tensors
    model_kwargs.pop("past_key_values", None)

  return (
    cur_len,
    next_running_sequences,
    next_running_scores,
    next_running_beam_indices,
    next_sequences,
    next_scores,
    next_beam_indices,
    next_is_sent_finished,
    next_model_kwargs,
  )


# 4. define "xla-compile-able" stop-condition and auto-regressive function
# define stop-condition and auto-regressive function
def beam_search_cond_fn(
        cur_len,
        running_sequences,
        running_scores,
        running_beam_indices,
        sequences,
        scores,
        beam_indices,
        is_sent_finished,
        model_kwargs,
):
  """
  Beam Search termination condition function -- halts the generation loop if any of these conditions becomes
  False
  """
  # 1. is less than max length?
  not_max_length_yet = cur_len < max_length

  # 2. can the new beams still improve?
  # early_stopping == False -> apply heuristic = always get the best score from `cur_len`. See the discussion
  # below for more details.
  # https://github.com/huggingface/transformers/pull/20901#issuecomment-1369845565
  # early_stopping == "never" -> compute the best score from max_length or cur_len, depending on the sign of
  #   length_penalty. Positive length_penalty favors longer sequences, thus we use max_length there.
  if early_stopping == "never" and length_penalty > 0.0:
    best_running_score = running_scores[:, :1] / (max_length ** length_penalty)
  else:
    best_running_score = running_scores[:, :1] / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
  worst_finished_score = tf.where(
    is_sent_finished, tf.math.reduce_min(scores, axis=1, keepdims=True), -1.0e9
  )
  improvement_still_possible = tf.math.reduce_any(best_running_score > worst_finished_score)

  # 3. is there still a beam that has not finished?
  still_open_beam = ~(tf.math.reduce_all(is_sent_finished) & (early_stopping is True))

  return not_max_length_yet & still_open_beam & improvement_still_possible



# 2-to-n generation steps can then be run in autoregressive fashion (only in case 1st generation step does
# NOT yield EOS token though)
# 5. run generation
# 1st generation step has to be run before to initialize `past_key_values` (if active)
(
  cur_len,
  running_sequences,
  running_scores,
  running_beam_indices,
  sequences,
  scores,
  beam_indices,
  is_sent_finished,
  model_kwargs,
) = beam_search_body_fn(
  cur_len,
  running_sequences,
  running_scores,
  running_beam_indices,
  sequences,
  scores,
  beam_indices,
  is_sent_finished,
  model_kwargs,
)

maximum_iterations = max_length - cur_len
(
  cur_len,
  running_sequences,
  running_scores,
  running_beam_indices,
  sequences,
  scores,
  beam_indices,
  is_sent_finished,
  _,
) = tf.while_loop(
  beam_search_cond_fn,
  beam_search_body_fn,
  (
    cur_len,
    running_sequences,
    running_scores,
    running_beam_indices,
    sequences,
    scores,
    beam_indices,
    is_sent_finished,
    model_kwargs,
  ),
  maximum_iterations=maximum_iterations,
)







# 6. prepare outputs
# Account for the edge-case where there are no finished sequences for a particular batch item. If so, return
# running sequences for that batch item.
none_finished = tf.math.reduce_any(is_sent_finished, axis=1)
sequences = tf.where(none_finished[:, None, None], sequences, running_sequences)
beam_indices = tf.where(none_finished[:, None, None], beam_indices, running_beam_indices)

# Apply the length penalty so that running scores match the finalized scores if they are used
running_scores = running_scores / (tf.cast(cur_len, dtype=tf.float32) ** length_penalty)
scores = tf.where(none_finished[:, None], scores, running_scores)

# Take best beams for each batch (the score is sorted in descending order)
sequences = flatten_beam_dim(sequences[:, :num_return_sequences, :])
scores = flatten_beam_dim(scores[:, :num_return_sequences])
beam_indices = flatten_beam_dim(beam_indices[:, :num_return_sequences, :])

if not use_xla:
  # Cut for backward compatibility
  sequences = sequences[:, :cur_len]
  beam_indices = beam_indices[:, :cur_len]

print(sequences)
