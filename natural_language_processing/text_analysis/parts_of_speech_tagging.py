# Before running this program run the following on the conda environment to download en_core_web_sm that is linked as en
# python -m spacy download en_core_web_sm
import spacy

nlp = spacy.load('en_core_web_sm')
text = nlp(u'We catched fish, and talked, and we took a swim now and then to keep off sleepiness')

for token in text:
    print(token.text, token.lemma_, token.pos_, token.tag_, token.dep_, token.shape_, token.is_alpha, token.is_stop)


