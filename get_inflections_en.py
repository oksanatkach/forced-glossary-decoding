
import spacy
import pyinflect

# nlp = spacy.load('en_core_web_sm')
# doc = nlp(glossary_term['en'])
# print(doc[0].tag_)
# print(doc[0]._.inflect('VBD', form_num=2))
all_forms_resp = pyinflect.getAllInflections(glossary_term['en'], pos_type=glossary_term['pos'])
all_forms = {all_forms_resp[k][0] for k in all_forms_resp}
print(all_forms)
