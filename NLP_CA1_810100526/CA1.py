#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 11 11:29:11 2022

@author: mac
"""
########## part1 #############

corpus = "low low low low low lower lower widest widest widest newest newest newest newest newest"

class BPE_Tokenizer:
    
    def __init__(self,vocab_size):
        self.dictionary = {}
        self.vocab_size = vocab_size
        
    def create_dictionary(self,corpus):
        #corpus = corpus.replace(" ","_ ")
        words = corpus.split()
        for word in words:
            key = " ".join(word)
            current_freq = self.dictionary.get(key, 0)
            self.dictionary[key] = 1+current_freq
        return self.dictionary
    
    def get_best_pair(self):
        pairs = {}
        for key,freq in self.dictionary.items():
            splited = key.split()
            for j in range(0,len(splited)-1):
                pair = (splited[j],splited[j+1])
                current_freq = pairs.get(pair, 0)
                pairs[pair] = current_freq+freq
        print("pairs:",pairs)
        return max(pairs, key=pairs.get)
    
    def merge_dictionary(self,pair):
        best_pair_new = ''.join(pair)
        best_pair_old = ' '.join(pair)
        dictionary = list(self.dictionary.items())
        self.dictionary.clear()
        for old_key,freq in dictionary:
            new_key = old_key.replace(best_pair_old,best_pair_new)
            self.dictionary[new_key] = freq
        del dictionary
        print("dictionay:",self.dictionary)
        
    def generate_vocab(self,corpus):
        self.create_dictionary(corpus)
        self.vocab = list(set(corpus.replace(" ","")))
        for i in range(self.vocab_size):
            try:
                print('------------- iteration',i,'-------------\n')
                best_pair = self.get_best_pair()
                print("best pair: ",''.join(best_pair))
                self.merge_dictionary(best_pair)
                self.vocab.append(''.join(best_pair))
                print("vocab:",self.vocab)
                print("\n")
            except:
                break
        return self.vocab
    
    def merge_word(self,word,pair):
        merged_word = []
        idx = 0
        while(idx < len(word)):
            if(idx == len(word)-1):
                merged_word.append(word[idx])
                idx += 1
            elif(''.join((word[idx],word[idx+1])) == pair):
                merged_word.append(pair)
                idx += 2
            else:
                merged_word.append(word[idx])
                idx += 1
        return merged_word
    
    def tokenize(self,word):
        print("---------- tokenizing",word,'----------\n')
        splited_word = list(word)
        while True:
            stop_loop = True
            pairs = set()
            for j in range(0,len(splited_word)-1):
                pairs.add(''.join((splited_word[j], splited_word[j+1])))
            for vocab_pair in self.vocab:
                if(vocab_pair in pairs):
                    stop_loop = False
                    splited_word = self.merge_word(splited_word,vocab_pair)
                    print("("+vocab_pair+") - merged word: ",splited_word )
                    break
            if(stop_loop):
                break
                
        return splited_word
        
        
bpe = BPE_Tokenizer(12)
bpe.generate_vocab(corpus)
bpe.tokenize("lowest")

print("\n")
########## part2 #############
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from tokenizers.pre_tokenizers import Whitespace
import os

def apply_BPE(train_file,input,save_file):
    BPE_tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    BPE_trainer = BpeTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    BPE_tokenizer.pre_tokenizer = Whitespace()
    BPE_tokenizer.train(train_file, BPE_trainer)    
    BPE_tokenizer.save(save_file)   
    BPE_output = BPE_tokenizer.encode(input)
    output = BPE_output.tokens
    print("tokenize:",output)
    print("size:",len(output))
    print("\n")
    
def apply_WordPiece(train_file,input,save_file):
    WP_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    WP_trainer = WordPieceTrainer(special_tokens=["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"])
    WP_tokenizer.pre_tokenizer = Whitespace()
    WP_tokenizer.train(train_file, WP_trainer)    
    WP_tokenizer.save(save_file)   
    WP_output = WP_tokenizer.encode(input)
    output = WP_output.tokens
    print("tokenize:",output)
    print("size:",len(output))
    print("\n")
    

if not os.path.exists('tokenizer'):
    os.makedirs('tokenizer')
    
wiki_files = [f"data/wikitext-103-raw/wiki.{split}.raw" for split in ["test", "train", "valid"]]
guten_file = ["data/gutenberg.txt"]
input = "This is a deep learning tokenization tutorial. Tokenization is the first step in a deep learning NLP pipeline. We will be comparing the tokens generated by each tokenization model. Excited much?!😍"

print("-------------- wp-tokenizer-guten --------------")
apply_WordPiece(guten_file,input,"tokenizer/wp-guten.json")
print("-------------- wp-tokenizer-wiki --------------")
apply_WordPiece(wiki_files,input,"tokenizer/wp-wiki.json")
print("-------------- bpe-tokenizer-guten --------------")
apply_BPE(guten_file,input,"tokenizer/bpe-guten.json")
print("-------------- bpe-tokenizer-wiki --------------")
apply_BPE(wiki_files,input,"tokenizer/bpe-wiki.json")
########## part3 #############
with open ("data/gutenberg.txt", "r") as myfile:
    guten_input = myfile.read()

wp_guten_tokenizer = Tokenizer.from_file("tokenizer/wp-guten.json")
wp_wiki_tokenizer = Tokenizer.from_file("tokenizer/wp-wiki.json")
bpe_guten_tokenizer = Tokenizer.from_file("tokenizer/bpe-guten.json")
bpe_wiki_tokenizer = Tokenizer.from_file("tokenizer/bpe-wiki.json")

tokenizers = [wp_guten_tokenizer,wp_wiki_tokenizer,bpe_guten_tokenizer,bpe_wiki_tokenizer]
texts = ["wp-guten","wp-wiki","bpe-guten","bpe-wiki"]
for i in range(0,4):
    out = tokenizers[i].encode(guten_input).tokens
    print(texts[i],":",len(out))
    
    
    

          