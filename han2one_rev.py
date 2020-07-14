import numpy as np
from hgtk.letter import decompose as decom

choseng = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄲ','ㄸ','ㅃ','ㅆ','ㅉ']
cwungseng = ['ㅏ','ㅑ','ㅓ','ㅕ','ㅗ','ㅛ','ㅜ','ㅠ','ㅡ','ㅣ','ㅐ','ㅒ','ㅔ','ㅖ','ㅘ','ㅙ','ㅚ','ㅝ','ㅞ','ㅟ','ㅢ']
congseng = ['ㄱ','ㄴ','ㄷ','ㄹ','ㅁ','ㅂ','ㅅ','ㅇ','ㅈ','ㅊ','ㅋ','ㅌ','ㅍ','ㅎ','ㄲ','ㅆ','ㄳ','ㄵ','ㄶ','ㄺ','ㄻ','ㄼ','ㄽ','ㄾ','ㄿ','ㅀ','ㅄ','']
alp = choseng+cwungseng+congseng
uniquealp = list(set(choseng+cwungseng+congseng))

def cho2onehot(s):
    res = np.zeros(len(choseng))
    if s in choseng:
        res[choseng.index(s)]=1
    return res

def cwu2onehot(s):
    res = np.zeros(len(cwungseng))
    if s in cwungseng:
        res[cwungseng.index(s)]=1
    return res

def con2onehot(s):
    res = np.zeros(len(congseng))
    if s in congseng:
        res[congseng.index(s)]=1
    return res

def uni2onehot(s):
    res = np.zeros(len(uniquealp))
    if s in uniquealp:
        res[uniquealp.index(s)]=1
    return res

def shin_onehot(s):
    z = decom(s)
    res = np.zeros((len(alp),3))
    res[:len(choseng),0] = cho2onehot(z[0])
    res[len(choseng):len(choseng)+len(cwungseng),1] = cwu2onehot(z[1])
    res[len(choseng)+len(cwungseng):len(alp),2] = con2onehot(z[2])
    return res

def cho_onehot(s):
    z = decom(s)
    res = np.zeros((len(alp)+len(uniquealp),3))
    if len(z[0]+z[1]+z[2]) > 1:
        res[:len(alp),:] = shin_onehot(s)
    elif len(z[0])>0:
        res[len(alp):,0] = uni2onehot(s)
    elif len(z[1])>0:
        res[len(alp):,1] = uni2onehot(s)
    else:
        res[len(alp):,2] = uni2onehot(s)
    return res

def char2onehot(s):
    z = decom(s)
    res = np.concatenate([cho2onehot(z[0]),cwu2onehot(z[1]),con2onehot(z[2])])
    return res
