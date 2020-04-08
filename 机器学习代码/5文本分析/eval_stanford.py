# _*_coding:utf-8_*_
from stanfordcorenlp import StanfordCoreNLP
import sys, json

nlp = StanfordCoreNLP(r'/Users/lully/Documents/MSE/JingDong/Code/demo/stanford-corenlp-full-2017-06-09/',lang='zh',memory='8g')

sentence = '一只小猫抓老鼠'
print(sentence)
##print(nlp.segment(sentence))
tokens = nlp.word_tokenize(sentence)
print("tokens",tokens)
pos_tag = nlp.pos_tag(sentence)
print("pos_tag:",pos_tag)

#ner = nlp.ner(sentence)
#print(ner)
print("parse:",nlp.parse(sentence))
print("dependency_parse:")
dep =nlp.dependency_parse(sentence)
print(dep)
for s in dep:
    print(s[0],tokens[s[1]-1],tokens[s[2]-1])
##print("annotate:")
#res_str = nlp.annotate(" ".join(nlp.word_tokenize(sentence)))
#print(res_str)
#res = json.loads(res_str)
#key = ""
#dep_list = res["sentences"]["basicDependencies"]
#for d in dep_list:
#    if d["dep"] == "ROOT":
#        key = d["dependentGloss"]
#        break
#for d in dep_list:
#    if d["dep"] == "dep" or d["dep"] == "dobj":
#        print(key,d["dependentGloss"])

