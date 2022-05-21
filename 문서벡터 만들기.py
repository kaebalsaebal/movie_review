# Document vector: sparse matrix
# input : NSMC train/test file -- load_NSMCdata(filename)
# output: document vector for each movie review
import sklearn
import sys

from sklearn.feature_extraction.text import TfidfVectorizer


def load_NSMCdata(filename):
    with open(filename, 'r', encoding='utf8') as f:
        data = [line.rstrip().split('\t') for line in f.readlines()]
    data = data[1:]  # remove 1st line(header)

    corpus = []
    target = []
    for sent in data:
        corpus.append(' '.join(sent[1:-1]))  # ignore id and label
        target.append(sent[-1])  # label 0/1
    return corpus, target


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("C> test.py ratings_train-ASP-KMA-EUCKR.txt")
        exit()

    # f = open(sys.argv[1], 'r', encoding='utf-8')
    # corpus = f.readlines()
    X_train, y_train = load_NSMCdata(sys.argv[1]) # 원본 불러오기-ratings.txt
    print(X_train[:5])

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(X_train)

    print(vectorizer.get_feature_names_out()[-100:])
    print(X.shape)  # csr_matrix -- Compressed Sparse Row

    # TEST -- 코사인 유사도 계산
    # from sklearn.metrics.pairwise import cosine_similarity
    # v = X.toarray()		# memory 부족 문제 발생!
    # print("SIMcos(d1, d2) = ", cosine_similarity([v[0]], [v[1]]))

    print(f"X[0] =\n{X[0]}")
    print(f"X[1] =\n{X[1]}")

    i = 1		# i-th document
    print("csr_matrix:", len(X[i].indptr),
          len(X[i].indices), len(X[i].data))
    # print(X[i].indptr[j])

    ftrain = open(sys.argv[2], 'w', encoding='utf-8')  # 벡터 저장파일-15만개 - dv_ratings_train.txt
    ftest = open(sys.argv[3], 'w', encoding='utf8')  # 벡터 저장파일-5만개 - dv_ratings_test.txt
    for i in range(0, X.shape[0]):
        if i < 150000:
            f = ftrain
        else:
            f = ftest
        print(i)
        y_train[i] = -1 if y_train[i] == '0' else 1  # 라벨 0인거 -1로 변환

        f.write(f"{y_train[i]} ")

        temp = [(X[i].indices[j], X[i].data[j])
                for j in range(len(X[i].data))]
        temp.sort(key=lambda a: a[0])

        for j in range(len(temp)):
            # 문서벡터 출력
            # print(f"doc[{i}][{j}] = {temp[j][0]}:{temp[j][1]}")
            # 문서벡터 작성
            f.write(f"{temp[j][0]+1}:{temp[j][1]} ")
        f.write('\n')

    # id순으로 sorting하여 SVM example1 데이터 포맷으로 파일에 저장!
    # 이 파일에 저장하는 부분만 추가하면 기말과제 2번의 2)의 문서벡터 15+5만개 생성 완료
    # target[i] -- positive/negative label
