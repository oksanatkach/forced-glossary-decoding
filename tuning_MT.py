from transformers import AutoTokenizer, MarianMTModel

strings = [
    ('Tap the screen anywhere', 'Торкніться екрана будь-де'),
    ('Tap here to continue', 'Торкніться тут, щоб продовжити'),
    ('tap', 'Торкнутись'),
    # ('he tapped out', 'він здався'),
    ('To continue, tap the button at the top of the screen', 'Щоб продовжити, торкніться кнопки зверху екрана')
]
model_name = "Helsinki-NLP/opus-mt-en-uk"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = MarianMTModel.from_pretrained(model_name)

processed = tokenizer('You are my friend', return_tensors="pt", padding=True)
force_words_ids = tokenizer(text_target=['Ви'], add_special_tokens=False).input_ids
outputs = model.generate(
    processed.input_ids,
    force_words_ids=force_words_ids,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)
print([tokenizer.decode(t, skip_special_tokens=True) for t in outputs][0])

processed = tokenizer('I would like to inform you', return_tensors="pt", padding=True)
force_words_ids = tokenizer(text_target=['тебе'], add_special_tokens=False).input_ids
outputs = model.generate(
    processed.input_ids,
    force_words_ids=force_words_ids,
    num_beams=5,
    num_return_sequences=1,
    no_repeat_ngram_size=1,
    remove_invalid_values=True,
)
print([tokenizer.decode(t, skip_special_tokens=True) for t in outputs][0])




# force_words_ids = tokenizer(text_target=['торкніться'], add_special_tokens=False).input_ids

# for source, target in strings:
#     processed = tokenizer(source, return_tensors="pt", padding=True)
#     outputs = model.generate(
#         processed.input_ids,
#         force_words_ids=force_words_ids,
#         num_beams=5,
#         num_return_sequences=1,
#         no_repeat_ngram_size=1,
#         remove_invalid_values=True,
#     )
#     # translated = model.generate(**processed)
#     print([tokenizer.decode(t, skip_special_tokens=True) for t in outputs][0])
#     print(target)
#     print('################')






# no_logits_model = model.model
# processed = tokenizer(strings[4][0], return_tensors="tf", padding=True)
# processed = tokenizer('To continue, tap the button at the top of the screen', return_tensors="tf", padding=True)
# print(processed['input_ids'])
# outputs = no_logits_model(**processed)
# print(outputs)
# model = TFAutoModelForSeq2SeqLM.from_pretrained(model_name)
# print(model.summary())
# print(model.config)

# from transformers import TrainingArguments

# training_args = TrainingArguments(output_dir="test_trainer")
# print(training_args)
# translated = model.generate(**tokenizer([tpl[0] for tpl in strings], return_tensors="tf", padding=True))
# print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
# print(processed)
# translated = model.generate(**processed)
# translated = model.generate(**tokenizer(strings[1][0], return_tensors="tf", padding=True))
# print(translated)
# print([tokenizer.decode(t, skip_special_tokens=True) for t in translated])
