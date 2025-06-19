from transformers import BertConfig, TFBertModel
import tensorflow as tf
from transformers import AutoTokenizer

checkpoint = "distilbert-base-uncased-finetuned-sst-2-english"
tokenizer = AutoTokenizer.from_pretrained(checkpoint)

model = TFBertModel.from_pretrained("bert-base-cased")

strings = [
    ('Tap the screen anywhere', 'Торкніться екрана будь-де'),
    # ('Tap here to continue', 'Торкніться тут, щоб продовжити'),
    ('tap', 'Торкнутись'),
    ('he tapped out', 'він здався'),
    ('To continue, tap the button at the top of the screen', 'торкніться кнопки зверху екрана')
]
en_strings = [tpl[0] for tpl in strings]
model_inputs = tokenizer(en_strings, padding=True, return_tensors="tf")
# for el in model_inputs['input_ids']:
#     print(el)
output = model(**model_inputs)
output = output.values()
output = list(output)
word_vectors = output[0]
sentence_vectors = output[1]
# print(word_vectors[0])

tap_1 = word_vectors[0][1]
tap_2 = word_vectors[1][1]
tap_3 = word_vectors[2][2]
tap_4 = word_vectors[3][4]
tap_vectors = [tap_1, tap_2, tap_3, tap_4]

tap_vectors_range = list(range(len(tap_vectors)))

for ind in tap_vectors_range:
    print(ind)
    print('########')
    for ind_2 in tap_vectors_range:
        if ind != ind_2:
            print(ind_2)
            print(tf.tensordot(tap_vectors[ind], tap_vectors[ind_2], 1))
    print('##################################')



# sequences = ["Hello!", "Cool.", "Nice!"]
# encoded_sequences = [
#     [101, 7592, 999, 102],
#     [101, 4658, 1012, 102],
#     [101, 3835, 999, 102],
# ]

# model_inputs = tf.constant(encoded_sequences)
# output = model(model_inputs)
# print(output)
