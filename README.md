# SMART-G2P
This repository is the official implementation of SMART-G2P

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

- Optional (for G2P process)
```
git clone https://github.com/scarletcho/KoG2P
pip install https://github.com/Kyubyong/g2pK
```

## HowTo
In python console:
```
>>> from sentranslit import sentranslit as trans
>>> trans('1999년 8월29일은 john가 mary을 만난 날로 매년 3시15분 방 3-147에서 의식이 거행된다')
'천구백구십구년 팔월이십구일은 존이 메리를 만난 날로 매년 세시십오분 방 삼다시일사칠에서 의식이 거행된다'
```
- Optional

If you installed *KoG2P* or *g2pK*, then uncomment either of the followings in *sentranslit.py*:
```python
## The default setting is g2pK, which is set along with this package
from g2p import G2p
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
We sincerely thank [Muik Jeon](https://github.com/muik) for letting us utilize the dataset from [transliteration](https://github.com/muik/transliteration) repository. 

## ToDo
- Enhance DL-based en-ko transliteration (via Transformer?)
- Elaborate number and symbol readings in Korean context
- Add corpus-level processing function
- Develop new DL-based G2P that considers polysemy
