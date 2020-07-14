import hgtk
import hanja
from mecab import Tagger
tagger = Tagger()

from predict import return_trans
from utils import decide_acronym, read_acronym
from utils import readNumberKor, readNumberEng, readNumber, readBigNum, readOnlyNum
from utils import dataset, small, big, real_latin
from utils import puncs, symbols, sym_han, sym_pro, count_symbols, count_sym_han
import re

## Dictionary from https://github.com/muik/transliteration/tree/master/data/source
dataset = dataset()
data_dict = {re.sub(' +', ' ',dataset[i][0]).lower(): re.sub(' +', ' ',dataset[i][1]) for i in range(len(dataset))}

def align_particles(sentence):
    s = sentence.split()
    particles = tagger.parse(sentence)
    chunks = []
    final  = False
    if len(particles) > 0:
        count_word = 0
        morphemes = []
        total = []
        for i in range(len(particles)):
            morphemes.append(particles[i][0])
            total.append(particles[i])
            if i+1 < len(particles):
                morphemes_temp = morphemes[:]
                morphemes_temp.append(particles[i+1][0])
                if "".join(morphemes_temp) not in s[count_word]:
                    chunks.append(total)
                    count_word += 1
                    morphemes = []
                    total = []
            else:
                chunks.append(total)
    return s, particles, chunks

def info_to_word(chunks):
    res = []
    for i in range(len(chunks)):
        temp = []
        for j in range(len(chunks[i])):
            temp.append(chunks[i][j][0])
        res.append(temp)
    return res

def trans_number(n,prev_term,next_term): ## Context-given number reading
    if hgtk.checker.is_hangul(prev_term) and hgtk.checker.is_hangul(next_term):
        return readNumberKor(n,next_term)
    elif real_latin(prev_term) or real_latin(next_term):
        if hgtk.checker.is_hangul(next_term) and n>10:
            return readNumberKor(n,next_term)
        else:
            return readNumberEng(n)
    else: ## Maybe hanja
        if prev_term in symbols or next_term in symbols:
            return readOnlyNum(n)
        elif n > 99999:
            return readBigNum(n)
        else:
            return readNumber(n)

def trans_symbol(symbol,prev_term,next_term):
    if symbol in count_symbols:
        return count_sym_han[count_symbols.index(symbol)]
    elif prev_term not in puncs:
        if hgtk.checker.is_hangul(prev_term) or hgtk.checker.is_hangul(next_term):
            return sym_han[symbols.index(symbol)]
        elif prev_term.isdigit() or next_term.isdigit():
            return sym_han[symbols.index(symbol)]
        elif real_latin(prev_term) or real_latin(next_term):
            return sym_pro[symbols.index(symbol)]
        else:
            return ''
    else:
        return ''

def trans_hanja(term): ## Complementary check
    return hanja.translate(term,'substitution')

def trans_latin(term): ## Rule and training hybrid transliteration
    if term.lower() in data_dict:
        return data_dict[term.lower()]
    else:
        if decide_acronym(term):
            return read_acronym(term)
        else:
            return return_trans(term) ## Tentative

def decide_context(term,chunks,eojeol,i,j):
    if len(chunks) == 1: ## Only one eojeol
        if len(eojeol) == 1: ## Eojeol has a single morpheme
            return readNumber(term)
        else: ## Multiple morphemes
            if j == len(eojeol)-1:
                return chunks[i][j-1],chunks[i][j-1]
            elif j == 0:
                return chunks[i][j+1],chunks[i][j+1]
            else:
                return chunks[i][j-1],chunks[i][j+1]
    else: ## Multiple eojeols
        if len(eojeol) == 1: ## Eojeol has a single morpheme
            if i == len(chunks)-1:
                return chunks[i-1][-1],chunks[i-1][-1]
            elif i == 0:
                return chunks[i+1][0],chunks[i+1][0]
            else:
                return chunks[i-1][-1],chunks[i+1][0]
        else: ## Multiple morphemes
            if j == len(eojeol)-1:
                if i == len(chunks)-1: ## Truly last morpheme
                    return chunks[i][j-1],chunks[i][j-1]
                else:
                    return chunks[i][j-1],chunks[i+1][0]
            elif j == 0:
                if i == 0: ## Truly first morpheme
                    return chunks[i][j+1],chunks[i][j+1]
                else:
                    return chunks[i-1][-1],chunks[i][j+1]
            else:
                return chunks[i][j-1],chunks[i][j+1]

def trans_eojeol(chunks,chunks_4num,metadata,if_num=True,if_sym=True,if_han=True,if_eng=True,if_puncs=True,if_else=True):
    for i in range(len(chunks)):
        eojeol = chunks[i]
        for j in range(len(eojeol)):
            term = eojeol[j]
            if term.isdigit():
                if if_num:
                    term = int(term)
                    x,y = decide_context(term,chunks_4num,eojeol,i,j)
                    chunks[i][j] = trans_number(term,x,y) ## Reflects context
                else:
                    chunks[i][j] = term
            elif term in symbols+count_symbols and i+j>0: ## Symbols not sentence-first
                if if_sym:
                    x,y = decide_context(term,chunks_4num,eojeol,i,j)
                    chunks[i][j] = trans_symbol(term,x,y) ## Currently bypassing
                else:
                    chunks[i][j] = term
            elif hgtk.checker.is_hanja(term):
                if if_han:
                    chunks[i][j] = trans_hanja(term) ## Double check
                else:
                    chunks[i][j] = term
            elif real_latin(term):
                if if_eng:
                    chunks[i][j] = trans_latin(term) ## Transliteration (or bypassing)
                else:
                    chunks[i][j] = term
            elif term in puncs:
                if if_puncs:
                    chunks[i][j] = term ## Bypassing by default
                else:
                    chunks[i][j] = ''
            elif hgtk.checker.is_hangul(term):
                chunks[i][j] = term ## Bypassing by default
            else:
                if if_else:
                    chunks[i][j] = term # '' ## Currently bypassing but able to delete
                else:
                    chunks[i][j] = ''
    return chunks

josa_o = ['은','이','과','을','이다']
josa_x = ['는','가','와','를','다']

def decide_josa(context,term):
    if hgtk.checker.is_hangul(context):
        dec = (hgtk.letter.decompose(context[-1])[2] != '') # If third sound is non-empty
        if term in josa_o and not dec:
            return josa_x[josa_o.index(term)]
        elif term in josa_x and dec:
            return josa_o[josa_x.index(term)]
        else:
            return term
    else:
        return term

def check_josa(chunks,chunks_4num,metadata):
    for i in range(len(chunks)):
        eojeol = chunks[i]
        for j in range(len(eojeol)):
            term = eojeol[j]
            pos = metadata[i][j][1].split(',')[0].lower()
            if pos[0] == 'j' and (term in josa_o or term in josa_x): # If pos is functional particle
                if j > 0 and chunks[i][j-1] != chunks_4num[i][j-1]:
                    chunks[i][j] = decide_josa(chunks[i][j-1],term)
                if i > 0 and j == 0 and chunks[i-1][-1] != chunks_4num[i-1][-1]:
                    chunks[i][j] = decide_josa(chunks[i-1][-1],term)
    return chunks

def leftword(chunks):
    for i in range(len(chunks)):
        eojeol = chunks[i]
        for j in range(len(eojeol)):
            term = chunks[i][j]
            if real_latin(term):
                chunks[i][j] = read_acronym(term)
            elif not hgtk.checker.is_hangul(term) and term not in puncs:
                chunks[i][j] = ''
    return chunks

def sentranslit(sentence,if_num=True,if_sym=True,if_han=True,if_eng=True,if_puncs=True,if_else=True):
    if if_han:
        sentence = hanja.translate(sentence,'substitution') ## For word-initial rule
    if not hgtk.checker.is_hangul(sentence): ## Only if contains non-Hangul terms
        s, particles, metadata  = align_particles(sentence)
        chunks = info_to_word(metadata)
        chunks_4num = info_to_word(metadata)
        mod_chunks = trans_eojeol(chunks,chunks_4num,metadata,if_num,if_sym,if_han,if_eng,if_puncs,if_else)   ## Chunks > Mod_chunks
        mod_chunks  = check_josa(mod_chunks,chunks_4num,metadata) ## Mod_chunks > Mod_final
        return (' ').join([''.join(z) for z in mod_chunks])
    else:
        return sentence

''' ## If KoG2P directory is cloned
from KoG2P.g2p import runKoG2P
'''

''' ## If G2pK is successfully installed
from g2p import G2p
g2p = G2p()
'''

def mixed_g2p(sentence,out_type='eng'):
    if out_type == 'kor':
        return g2p(sentence)
    else:
        return runKoG2P(sentence,'KoG2P/rulebook.txt')

