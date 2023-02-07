import glob
import string
import re
import hgtk
import numpy as np

def read_data(filename):
    with open(filename, 'r') as f:
        data = [line.split('\t') for line in f.read().splitlines()]
    return data

##### Making up the dictionary for the transliteration; builds dataset()

data = []
for name in glob.glob('transliteration/data/source/*'):
    data.append(read_data(name)[3:])

def dataset():
    dataset = []
    for i in range(len(data)):
        dataset += data[i]
    return dataset

##### Punctuations and symbols

puncs = ['?', '!', '.', ',', '~']
symbols = ['@', '#', '*', '(', ')', '+', '-', ';', ':', '/', '=', '&', '_', "'", '"']
sym_han = ['골뱅이', '샵', '별표', '괄호열고', '괄호닫고', '더하기', '다시', '세미콜론', '땡땡', '짝대기', '는', '그리고', '밑줄', '따옴표', '쌍따옴표']
sym_pro = ['앳', '넘버', '스타', '괄호열고', '괄호닫고', '플러스', '대쉬', '세미콜론', '콜론', '슬래쉬', '이퀄스', '앤드', '언더바', '어퍼스트로피', '쌍따옴표']
count_symbols = ['$', '￦', '￡', '￥', '€', '℃', '%']
count_sym_han = ['달러', '원', '파운드', '엔', '유로', '도씨', '퍼센트']

##### Utils on English reading

small = {}
for i in range(len(string.ascii_lowercase)):
    small.update({string.ascii_lowercase[i]:i})

big = {}
for i in range(len(string.ascii_uppercase)):
    big.update({string.ascii_uppercase[i]:i+26})

## Decides if a term consists of latin alphabet
def real_latin(term):
    if hgtk.checker.is_latin1(term) and (term[0] in small or term[0] in big):
        return True
    else:
        return False

vowels = ['a','e','i','o','u']
alpha  = ['에이','비','씨','디','이','에프','쥐','에이치','아이','제이','케이','엘','엠','엔','오','피','큐','알','에스','티','유','브이','더블유','엑스','와이','지']

## Decides if a term is acronym
def decide_acronym(term):
    if len(term) == 1:
        return True
    elif sum([int((z in big)) for z in term]) == len(term):
        return True
    elif sum([int((z.lower() in vowels)) for z in term]) == 0:
        return True
    else:
        return False

## Reads English acronym
def read_acronym(term):
    return ('').join([alpha[int(small[z.lower()])] for z in term])

##### Utils on number reading in Korean and English

kor_num0 = ['','하나','둘','셋','넷','다섯','여섯','일곱','여덟','아홉']
kor_num1 = ['열','스물','서른','마흔','쉰','예순','일흔','여든','아흔']
kor_cnt0 = ['','한','두','세','네','다섯','여섯','일곱','여덟','아홉']

count_noun = ['개','번째','번','살','시','걸음'] # To be supplemented
bbong_noun = ['이상','이하','초과','미만'] # To be supplemented

## Count numbers in order
def makeCountKor(n):
    x, y = divmod(n,10)
    if x < 1:
        if y == 0:
            return '영'
        else:
            return kor_cnt0[y]
    else:
        return kor_num1[x-1]+kor_cnt0[y]

## Count numbers in nominal form
def makeBbongKor(n):
    x = int(np.floor(n/10))
    y = int(n - 10*x)
    if x < 1:
        if y == 0:
            return '영'
        else:
            return kor_num0[y]
    else:
        return kor_num1[x-1]+kor_num0[y]

## Read numbers in Korean
def readNumberKor(n,meta):
    if meta in count_noun and n<100:
        return makeCountKor(n)
    elif meta in bbong_noun and n<100:
        #return makeBbongKor(n)
        return readNumber(n)
    else:
        return readNumber(n)

eng_num0 = ['','원','투','쓰리','포','파이브','식스','세븐','에잇','나인']
eng_num1 = ['텐','트웬티','써티','포티','피프티','식스티','세븐티','에잇티','나인티']
eng_numt = ['','일레븐','투웰브','써틴','포틴','피프틴','식스틴','세븐틴','에잇틴','나인틴']
eng_read = ['오','원','투','쓰리','포','파이브','식스','세븐','에잇','나인']

## Read numbers in English
def readNumberEng(n):
    x = int(np.floor(n/10))
    y = int(n - 10*x)
    if x < 1:
        return eng_num0[y]
    elif x < 10:
        if x < 2:
            if y == 0:
                return eng_num1[y]
            else:
                return eng_numt[y]
        else:
            return eng_num1[x-1]+eng_num0[y]
    else:
        seq = [eng_read[int(z)] for z in str(n)]
        return ('').join(seq)

## Read numbers in sino-Korean
## Refer to: https://soooprmx.tistory.com/entry/%ED%8C%8C%EC%9D%B4%EC%8D%AC-%EC%88%AB%EC%9E%90%EB%A5%BC-%ED%95%9C%EA%B8%80%EB%A1%9C-%EC%9D%BD%EB%8A%94-%ED%95%A8%EC%88%98
def readNumber(n):
    units = [''] + list('십백천만')
    nums = '일이삼사오육칠팔구'
    result = []
    i = 0
    while n > 0:
        n, r = divmod(n, 10)
        if r > 0:
            result.append(units[i])
            if r >= 1:
                result.append(nums[r-1])
        i += 1
    return ''.join(result[::-1])

## Read large numbers
def readBigNum(n):
    units = [''] + list('만억조경해자양구간정재극')
    nums = '일이삼사오육칠팔구'
    result = []
    i = 0
    while n > 0:
        n, r = divmod(n, 10000)
        if r > 0:
            result.append(readNumber(r)+units[i])
        i += 1
    return ''.join(result[::-1])

## Read numbers continuously
def readOnlyNum(n):
    nums = '영일이삼사오육칠팔구'
    read = [nums[int(z)] for z in str(n)]
    return ''.join(read)
