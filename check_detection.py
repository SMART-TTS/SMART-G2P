from eval import test_eng_corpus

score = 0
for i in range(10):
    score += test_eng_corpus('test/eng_test.txt',10)

print("AVERAGE SCORE:", score/10)
