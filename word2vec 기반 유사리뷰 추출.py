from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
import sys

def load_NSMCdata(filename):
	with open(filename, 'r', encoding='utf8') as f:
			data = [line.rstrip().split('\t') for line in f.readlines()]
	data = data[1:]	# remove 1st line(header)

	corpus = []
	target = []
	for sent in data:
			corpus.append(' '.join(sent[1:-1]))	# ignore id and label
			target.append(sent[-1])	# label 0/1
	return corpus, target
    
if __name__ == '__main__':
    X_train, y_train = load_NSMCdata(sys.argv[1]) # ratings.txt
    
    word2vec_model = Word2Vec(size = 300, window=5, min_count = 2, workers = -1)
    
    word2vec_model.build_vocab(X_train)
    word2vec_model.train(X_train, total_examples = word2vec_model.corpus_count, epochs=15)
    
    train = []
    test = []
    
    with open(sys.argv[1], 'r', encoding='utf8') as f:
        data = [line.rstrip() for line in f.readlines()]
        i=0
        for line in data:
            if i < 150000:
                a = train
            else:
                a = test
            doc2vec = None
            count=0
            for word in line.strip('\t'):
                if word in word2vec_model.wv.vocab:
                    count+=1
                    if doc2vec is None:
                        doc2vec = word2vec_model[word]
                    else:
                        doc2vec = doc2vec + word2vec_model[word]
            i+=1
            if doc2vec is not None:
                a.append(doc2vec)
                
    print(len(train),len(test))
    cossim = cosine_similarity(test,train)
    print(cossim.shape)
    
    ftrain = open(sys.argv[2], 'r', encoding='utf8') # ratings_train.txt
    ftest = open(sys.argv[3], 'r', encoding='utf8') # ratings_test.txt
    
    trainline = ftrain.readlines()[1:]
    testline = ftest.readlines()[1:]
    
    while True:
        num = int(input("문서를 입력하시오.\n"))
        
        result = list(enumerate(cossim[num]))
        result = sorted(result, key = lambda x: x[1], reverse = True)
        result = result[:5]
        
        print(testline[num-1])
        index = [i[0] for i in result]
        for i in index:
            print(trainline[i-1])