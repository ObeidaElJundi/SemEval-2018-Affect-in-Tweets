
# coding: utf-8

# # Preprocessing Arabic Text

# ### this module aims to preprocess arabic text (tweets specifically) in order to be efficiently used for Natural Language Processing & Machine Learning
# ### Preprocessing includes:
# - tokenization
# - remove mentions
# - normalize_hashtag: remove hashtag symbol (#) and replace underscores in hashtags by spaces
# - remove punctuations & diacritics
# - normalize (or standarize) Hamza & Ha2
# - remove repeating characters
# - remove english characters
# - remove stop words
# - replace emojies using manually crafted emojis lexicon
# - stemming

# ### import necessary libraries

# In[1]:


import numpy as np
import pandas as pd
import re
import nltk  # if nltk is being used for the first time, you may need to download some resources: nltk.download()
from nltk.stem.arlstem import ARLSTem  # requires minimum nltk version of 3.2.5
import string


# ### Defining Arabic preprocessing class

# In[2]:


class Arabic_preprocessing:
    
    
    def __init__(self):
        #preparing arabic emojis lexicon
        emojis_lexicon=pd.read_csv('resources/emoji_lexicon.csv',header=None, names=['emoji', 'utf-8','unicode','arabic_translation'])
        self.emojis_lexicon_dict = dict()
        for index, row in emojis_lexicon.iterrows():
            self.emojis_lexicon_dict[row['emoji'].replace(' ','')] = row['arabic_translation']

        #preparing arabic stop words list
        f=open('resources/arabic_stop_words.txt', 'r', encoding="utf8")  # read stop words from file
        self.stop_words = [line.strip() for line in f.readlines()]  # construct list of stop wrods and remove the new line character

        #preparing punctuations list
        arabic_punctuations = '''`÷×؛<>_()*&^%][ـ،/:"؟.,'{}~¦+|!”…“–ـ'''
        english_punctuations = string.punctuation
        self.all_punctuations = set(arabic_punctuations + english_punctuations)

        # initializing the stemmer
        self.stemmer = ARLSTem()  # requires minimum NLTK version of 3.2.5
        
        self.arabic_diacritics = re.compile("""
                                         ّ    | # Tashdid
                                         َ    | # Fatha
                                         ً    | # Tanwin Fath
                                         ُ    | # Damma
                                         ٌ    | # Tanwin Damm
                                         ِ    | # Kasra
                                         ٍ    | # Tanwin Kasr
                                         ْ    | # Sukun
                                         ـ     # Tatwil/Kashida

                                     """, re.VERBOSE)

        
    def normalize_arabic(self, text):
        text = re.sub("[إأآاٱ]", "ا", text)
        text = re.sub("ى", "ي", text)
        #text = re.sub("ؤ", "ء", text)
        #text = re.sub("ئ", "ء", text)
        text = re.sub("ة", "ه", text)  # replace ta2 marboota by ha2
        text = re.sub("گ", "ك", text)
        text = re.sub("\u0640", '', text)  # remove tatweel
        return text


    def remove_punctuations(self, text):
        for p in self.all_punctuations:
            if p in text:
                text = text.replace(p, '')
        return text


    def remove_diacritics(self, text):
        text = re.sub(self.arabic_diacritics, '', text)
        return text


    def remove_repeating_char(self, text):
        return re.sub(r'(.)\1+', r'\1', text)


    def remove_mention(self, text):
        return re.sub(r'@\S+', '', text)


    def hashtag_match(self, match_object):
        return match_object.group(1).replace('_', ' ')

    def normalize_hashtag(self, text):
        return re.sub(r'#(\S+)', self.hashtag_match, text)

    def emojis_match(self, match_object):
        return ' ' + ' '.join(list(match_object.group(1))) + ' '

    def separate_emojis(self, text):
        emojis_unicode = r'([\U0001F600-\U0001F64F\U00002000-\U00003000]+)'
        return re.sub(emojis_unicode, self.emojis_match, text)

    def replace_emojis(self, text):
        new_text = ""
        for l in text:
            new_text += self.emojis_lexicon_dict[l] if l in self.emojis_lexicon_dict.keys() else l
        return new_text

    def remove_english_characters(self, text):
        return re.sub(r'[a-zA-Z]+', '', text)
    
    def clean_stop_words(self):
        # normalize, and remove diacritics from, stop words to increase posibility of matching with normalized data
        self.stop_words = [self.remove_diacritics(self.normalize_arabic(word)) for word in self.stop_words]
    
    def preprocess_arabic_text(self, text, stem=True, replace_emojis=True, normalize_arabic=True):
        self.clean_stop_words()
        text = text.replace('\\n', ' ').replace('\n', ' ')
        text = self.remove_mention(text)
        text = self.normalize_hashtag(text)
        text = self.remove_punctuations(text)
        text = self.remove_diacritics(text)
        if normalize_arabic: text = self.normalize_arabic(text)
        text = self.separate_emojis(text)
        if replace_emojis: text = self.replace_emojis(text)
        text = self.remove_repeating_char(text)
        text = self.remove_english_characters(text)
        words = nltk.word_tokenize(text)
        words = [word for word in words if word not in self.stop_words]
        if stem: words = [self.stemmer.stem(word) for word in words]
        return ' '.join(words)  # return sentence (str), not list of words

