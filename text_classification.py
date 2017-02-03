import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import  numpy as np
import random
import pickle
from collections import Counter
from nltk.corpus import stopwords

from os import listdir
from os.path import isfile, join
import sys

reload(sys)
sys.setdefaultencoding('utf8')

classification = {}
lemmatizer = WordNetLemmatizer()
def get_list_of_files_from_directory(directory_path):
    # mypath = "./data/mini_newsgroups/alt.atheism"
    onlyfiles = [f for f in listdir(directory_path) if isfile(join(directory_path, f))]
    return onlyfiles

def create_lexicon(list_file_paths=[]):
    '''
    :param list_file_paths: Path of directory, from wich all the files have to be processed
    :return: returns a words from the files, after removing the stop words.
    '''
    lexicon = []
    for directory_path in list_file_paths:
        total_files_in_directory = get_list_of_files_from_directory(directory_path)
        for file_name in total_files_in_directory:
            with open(directory_path+"/"+file_name) as f:
                contents = f.readlines()
                # print "reading file",file_name,len(contents)
                for line in contents:
                    try:
                        all_words = word_tokenize(line)
                    except:
                        print directory_path+"/"+file_name
                        continue
                    # all_words = word_tokenize(line)
                    lexicon += list(all_words)

    #convert words like run, running => RUN
    tmp = []
    # numberOfwordsIgnored = 0
    for word in lexicon:
        # print word
        try:
            tmp.append(lemmatizer.lemmatize(word))
        except:
            print word
            continue
    # lexicon = [ lemmatizer.lemmatize(word) for word in lexicon]
    word_counts = Counter(tmp)
    # lexicon = [ lemmatizer.lemmatize(word) for word in lexicon]
    # word_counts = Counter(lexicon)
    l2 = []
    for w in word_counts:
        # if w not in stopwords.words("english"):
        #     l2.append(w)
        if 1000 > word_counts[w] > 50:
            l2.append(w)
        #     print w,word_counts[w]
    print len(l2)

    return l2


# hm_lines = 100000

def sample_handling(list_file_directories, lexicon, classification):
    featureset = []
    for directory_path in list_file_directories:
        total_files_in_directory = get_list_of_files_from_directory(directory_path)
        for file_name in total_files_in_directory:
            with open(directory_path+"/"+file_name) as f:
                contents = f.readlines()
                for line in contents:
                    try:
                        current_words = word_tokenize(line)
                    except:
                        print "ignoredd"
                        continue
                    tmp = []
                    for i in current_words:

                        try:
                            tmp.append(lemmatizer.lemmatize(i))
                        except:
                            print i
                            continue
                    current_words = tmp
                    # current_words = [lemmatizer.lemmatize(i) for i in current_words]
                    features = np.zeros(len(lexicon))
                    for word in current_words:
                        if word in lexicon:
                            index_value = lexicon.index(word)
                            features[index_value] += 1

                    features = list(features)
                    featureset.append([features, classification])
                return featureset

def create_feature_sets_and_labels(directory_paths, test_size = 0.1):
    '''

    :param directory_paths: ["directoryPath1", "directoryPath2"]
    :param test_size:
    :return:
    '''
    lexicon = create_lexicon(directory_paths)
    features = []
    # features += sample_handling("./data/pos.txt", lexicon, [1,0])
    # features += sample_handling("./data/neg.txt", lexicon, [0, 1])
    tmp = [[0,0,0],[0,0,1],[0,1,0]]
    number_of_classification = len(directory_paths)
    for i,directory in enumerate(directory_paths):
        directoryName = directory.split("/")[-1]
        classification_id = bin(i)[2:]
        classification[directoryName] = list(classification_id.zfill(number_of_classification))#classification_id if len(classification_id) >1 else "0"+classification_id
        features += sample_handling(directory,lexicon,list(classification[directoryName]))

    random.shuffle(features)

    features = np.array(features)
    testing_size = int(test_size*len(features))
    train_x = list(features[:,0][:-testing_size])
    '''
    sample input:
    [[5,8]
     [7,9]
    ]
    sample output:
    [5,7]
    '''
    train_y = list(features[:,1][:-testing_size])

    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    return train_x,train_y,test_x,test_y,classification

if __name__ == "__main__":

    print "started"
    directory_paths = ["./data/mini_newsgroups/alt.atheism","./data/mini_newsgroups/rec.autos"]
    train_x, train_y, test_x, test_y = create_feature_sets_and_labels(directory_paths)
    with open('./data/sentiment_set.pickle', "wb") as f:
        pickle.dump([train_x, train_y, test_x, test_y], f)

    # create_lexicon(["./data/mini_newsgroups/alt.atheism","./data/mini_newsgroups/rec.autos"])
    print "completed"

