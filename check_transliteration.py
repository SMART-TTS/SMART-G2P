from eval import test_trans_corpus

score = 0
for i in range(10):
    score += test_trans_corpus('test/trans_test.txt',10)

print("AVERAGE SCORE:", score/10)

