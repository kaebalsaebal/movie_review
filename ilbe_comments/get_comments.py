#-*- coding: utf-8 -*-
# Extract 1,196,345 comments to a text file
# Total 8,320,881 tokens: unique 2,299,252 tokens
# C> iconv -c -f utf-8 -t cp949 ilbe_comments_1.2m.txt > ilbe_comments_1.2m_cp949.txt

f1 = open("ilbe_comments_1.2m.tsv", "r", encoding="utf-8")
f2 = open("ilbe_comments_1.2m.txt", "w", encoding="utf-8")

while True:
    line = f1.readline()
    if not line: break
    f2.write(line.split('\t')[3])

f1.close()
f2.close()

