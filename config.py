from transformers import TFMarianModel, MarianConfig

# Initializing a Marian Helsinki-NLP/opus-mt-en-de style configuration
configuration = MarianConfig()

# Initializing a model from the Helsinki-NLP/opus-mt-en-de style configuration
model = TFMarianModel(configuration)

# Accessing the model configuration
configuration = model.config
print(configuration)
