from transformers import AutoTokenizer, TFMarianMTModel, TFAutoModelForSeq2SeqLM
from transformers import MarianMTModel
# import tensorflow as tf

model_name = "Helsinki-NLP/opus-mt-en-uk"
tokenizer = AutoTokenizer.from_pretrained(model_name)
# model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

# vector = tf.constant([[52264, 4, 3831, 6345, 0]], dtype=tf.int32)
# # print(tokenizer.decode(61586))
# # print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
# tokens = [61586, 1750, 6209, 4041, 10736, 26, 577, 63, 935, 2181, 0]
# for i in tokens:
#     print(tokenizer.decode(i))
#
#
# # tokens_2 = tokenizer('торкніться', return_tensors="tf", padding=True)['input_ids']
# tokens_2 = tokenizer('екран', return_tensors="tf")['input_ids']
# print(tokens_2)
# for i in [51198, 202]:
# for i in [24818, 9537]:
# for i in [1410,1062,2,5156,329]:
#     print(tokenizer.decode(i))
# print('#############################')
# print(tokenizer.decode(61586))

# vocab_weights = model.model.shared.weights[0]
# print(vocab_weights[24818])

# encoder_input_str = 'tap here to continue'
encoder_input_str = 'To continue, tap the button'
input_ids = tokenizer(encoder_input_str, return_tensors="pt").input_ids
# force_words_ids = tokenizer(text_target=['натисніть'], add_special_tokens=False).input_ids
force_words_ids = tokenizer(text_target=['торкніться'], add_special_tokens=False).input_ids
# print(force_words_ids)
outputs = model.generate(
    input_ids,
    # force_words_ids=force_words_ids,
    # num_beams=5,
    # num_return_sequences=1,
    # no_repeat_ngram_size=1,
    # remove_invalid_values=True,
)
print(outputs[0])
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
print(tokenizer.decode([0], skip_special_tokens=True))

# print(tokenizer(['here']))
# print(tokenizer(text_target=['торкніться']))
# tokens = [  790,   846,   265,  5265,  7815,  7593, 21978,   960,   505,    98]
# for tok in tokens:
#     print(tokenizer.decode(tok))
