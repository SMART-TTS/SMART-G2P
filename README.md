# SMART-G2P

SMART-G2P는 code-mixed G2P를 위한 한국어 발음변환 모듈로, 2021년도 과학기술통신부의 재원으로 정보통신기획평가원(IITP)의 지원을 받아 수행한 "소량 데이터만을 이용한 고품질 종단형 기반의 딥러닝 다화자 운율 및 감정 복제 기술 개발" 과제의 일환으로 공개된 코드입니다. 본 모델의 특징은 다음과 같습니다.

- 영어, 한문 등을 포함한 한국어 문장에 대해 해당 표현들을 한국어 발음으로 변경
- 추가적인 라이브러리 ([KoG2P](https://github.com/scarletcho/KoG2P) 혹은 [g2pK](https://github.com/Kyubyong/g2pK)) 활용을 통해, 한국어 문장을 발음열로 변환
- 영어, 한문, 숫자, 특수기호 등을 포함한 문장에서 선택적으로 한국어 발음 변경 여부를 결정

This repository is the official implementation of SMART-G2P, where the development of the algorithm and code was supported by IITP. Our module has the following features:

- Change the English and Kanji expressions in Korean sentence into Korean pronunciation
- Transform the grapheme sequence to phoneme sequence with the help of other G2P libraries (e.g., KoG2P, g2pK)
- Decide whether to transliterate specific expressions (English, Kanji, numbers, and special symbols) upon user's choice

## Environment
Under Python 3.6.6

## Install
```
git clone https://github.com/SMART-TTS/SMART-G2P
cd SMART-G2P
pip install pyyaml
pip install -r Requirements.txt
git clone https://github.com/muik/transliteration
```
If you have difficulty in importing tagger in MeCab, execute the following:
```
bash <(curl -s https://raw.githubusercontent.com/konlpy/konlpy/master/scripts/mecab.sh)
```
If MeCab still matters, then follow the instruction of [this link](https://sosomemo.tistory.com/30)

#### Optional (for G2P process)
```
git clone https://github.com/scarletcho/KoG2P
pip install https://github.com/Kyubyong/g2pK
```

## How To
For GPU setting, please check the following in *infer.py* and set appropriate device in your local:
```python
os.environ["CUDA_VISIBLE_DEVICES"]='1'
```
In python console:
```
>>> from trans import sentranslit as trans
>>> trans('1999년 8월29일은 john가 mary을 만난 날로 매년 3시15분 방 3-147에서 의식이 거행된다')
'천구백구십구년 팔월이십구일은 존이 메리를 만난 날로 매년 세시십오분 방 삼다시일사칠에서 의식이 거행된다'
```
To decide whether to change the expressions with each attribute, you can add one of the following arguments
```python
if_num=False,   # to keep numbers
if_sym=False,   # to keep special symbols ['@', '#', '*', '(', ')', '+', '-', ';', ':', '/', '=', '&', '_', "'", '"'] + ['$', '￦', '￡', '￥', '€', '℃', '%']
if_han=False,   # to keep kanji
if_eng=False,   # to keep English expressions
if_puncs=False, # to keep punctuations ['?', '!', '.', ',', '~']
if_else=False   # to keep other exceptions   
```
For example, 
```
>>> from trans import sentranslit as trans
>>> trans('1999년 8월29일은 john가 mary을 만난 날로 매년 3시15분 방 3-147에서 의식이 거행된다', if_num=False, if_sym=False)
'1999년 8월29일은 존이 메리를 만난 날로 매년 3시15분 방 3-147에서 의식이 거행된다'
```
#### Optional

If you installed *KoG2P* or *g2pK*, then uncomment either of the followings in *sentranslit.py*:
```python
## The default setting is g2pK, which is set along with this package
from g2pk import G2p
g2p = G2p()

## If KoG2P directory is cloned
from KoG2P.g2p import runKoG2P
```
You can choose the format of the outcome, English alphabets or Korean characters, by choosing *KoG2P* and *g2pK* respectively. For example, the following yields *g2pK* result, which is the default setting. For *KoG2P* result, add *out_type = 'eng'*.
```
>>> mixed_g2p('1999년 8월29일은 john가 mary을 만난 날로 매년 3시15분 방 3-147에서 의식이 거행된다')
'천구백꾸십꾸년 파뤄리십꾸이른 조니 메리를 만난 날로 매년 세시시보분 방 삼다시일사치레서 의시기 거행된다'
```

## Acknowledgement
We sincerely thank [Muik Jeon](https://github.com/muik) for letting us utilize the dataset from [transliteration](https://github.com/muik/transliteration) repository. Also, we appreciate all the contributors of the open libraries we found essential in our implementation.

## To Do
- Elaborate number and symbol readings in Korean context
- Add corpus-level processing function
- Develop new DL-based G2P that considers polysemy

## Citation
If you have found our module useful, please consider citing the following [paper](https://www.aclweb.org/anthology/2020.calcs-1.9):
```
@inproceedings{cho-etal-2020-towards,
    title = "Towards an Efficient Code-Mixed Grapheme-to-Phoneme Conversion in an Agglutinative Language: A Case Study on To-{K}orean Transliteration",
    author = "Cho, Won Ik  and
      Kim, Seok Min  and
      Kim, Nam Soo",
    booktitle = "Proceedings of the The 4th Workshop on Computational Approaches to Code Switching",
    month = may,
    year = "2020",
    address = "Marseille, France",
    publisher = "European Language Resources Association",
    url = "https://www.aclweb.org/anthology/2020.calcs-1.9",
    pages = "65--70",
    abstract = "Code-mixed grapheme-to-phoneme (G2P) conversion is a crucial issue for modern speech recognition and synthesis task, but has been seldom investigated in sentence-level in literature. In this study, we construct a system that performs precise and efficient multi-stage code-mixed G2P conversion, for a less studied agglutinative language, Korean. The proposed system undertakes a sentence-level transliteration that is effective in the accurate processing of Korean text. We formulate the underlying philosophy that supports our approach and demonstrate how it fits with the contemporary document.",
    language = "English",
    ISBN = "979-10-95546-66-5",
}
```

## Technical Report

Specification of this implementation can be found in the [technical report](https://www.dropbox.com/s/fow2d0nk5x2d70f/SMART-G2P.pdf?dl=0).

본 프로젝트 관련 개선사항들에 대한 기술문서는 [여기](https://www.dropbox.com/s/fow2d0nk5x2d70f/SMART-G2P.pdf?dl=0)를 참고해 주세요.
