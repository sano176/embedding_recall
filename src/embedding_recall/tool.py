import numpy as np
import faiss
import pickle
import fasttext
import json
import math
from collections import defaultdict
from collections import OrderedDict
import rocksdb
import re


class FaissHandler(object):

    def __init__(self, faiss_filepath, rocksdb_filepath, fasttext_filepath, mode='w', d=100, nlist=1024, k=10):
        self.d = d
        self.nlist = nlist
        self.k = k
        self.faiss_filepath = faiss_filepath
        self.mode = mode

        # rocksdb
        options = rocksdb.Options(create_if_missing=True)
        self.db = rocksdb.DB(rocksdb_filepath, options)

        # embedding vector
        self.fasttext = fasttext.load_model(fasttext_filepath)

    def __enter__(self):
        assert self.mode == 'w' or self.mode == 'r'
        if self.mode == 'w':
            quantize = faiss.IndexFlatL2(self.d)  # the other index
            self.index = faiss.IndexIVFPQ(quantize, self.d, self.nlist, 4, 8)
        elif self.mode == 'r':
            self.index = faiss.read_index(self.faiss_filepath)
        return self

    def initialize(self):
        self.index = faiss.read_index(self.faiss_filepath)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode == 'w':
            faiss.write_index(self.index, self.faiss_filepath)

    # 根据文本进行查找
    def search_query(self, querys, k=10, nprobe=10):
        vectors = self.getQuerysVector(querys)
        result = self.search(vectors)
        res = []
        for query, ele in zip(querys, result):
            res.append((query, ele))
        return res

    # 根据向量检索
    def search(self, datas, k=10, nprobe=10):
        datas = np.array(datas).astype('float32')
        self.index.nprobe = nprobe
        D, I = self.index.search(datas, k)
        result = self._search_result(D, I)
        return result

    # 加工处理这些信息
    def _search_result(self, D, I):
        result = []
        for distance, index in zip(D.tolist(), I.tolist()):
            res = []
            for d, i in zip(distance, index):
                key = str(i).encode('utf-8')
                value = self.db.get(key)
                value = value.decode('utf-8')
                value = json.loads(value, encoding='utf-8')
                value['distance'] = d
                res.append(value)
            result.append(res)

        return result

    def add(self, datas, forwardIndexInfo, isTrain):
        datas = np.array(datas).astype('float32')
        if isTrain:
            self.index.train(datas)
        self.index.add(datas)
        self._addRocksDB(forwardIndexInfo)

    def add_with_ids(self, datas, ids, forwardIndexInfo, isTrain):

        datas = np.array(datas).astype('float32')
        ids = np.array(ids).astype('long')
        if isTrain:
            self.index.train(datas)
        self.index.add_with_ids(datas, ids)
        self._addRocksDB(forwardIndexInfo)

    def _addRocksDB(self, forwardIndexInfo):
        for key, value in forwardIndexInfo:
            key = str(key).encode('utf-8')
            value = json.dumps(value, ensure_ascii=False).encode('utf-8')
            self.db.put(key, value)

    def getQuerysVector(self, querys):
        result = []
        for query in querys:
            vector = self.fasttext.get_sentence_vector(query)
            result.append(vector)

        return result

    def get_sentence_vector(self, query):
        return self.fasttext.get_sentence_vector(query)

    # 确立中心点的东西
    def getEachCenterQuery(self, querys, k=3):

        vectors = getQuerysVector(querys)
        vectors = np.array(vectors).astype('float32')
        D, I = self.index.search(rvectors, k)

        statistic = defaultdict(list)
        for query, center, distance in zip(querys, I.tolist()[0], D.tolist()[0]):
            statistic[center].append((query, distance))

        temp = dict()
        for key, value in statistic.items():
            value = sorted(value, key=lambda x: x[1], reverse=False)
            temp[key] = value

        center_size = len(temp)

        modify_statistic = {}
        for key, value in temp.items():
            weight = len(value) / len(querys)
            temp = {}
            temp['weight'] = weight
            temp['querys'] = value[:3]
            modify_statistic[key] = temp

        return modify_statistic

    def shuffle_results(self, recommends):

        i = 0
        result = []
        mark = True
        while mark:
            temp_mark = False
            for key, value in recommends:
                if i < len(value):
                    temp_mark = True
                    result.append(value[i])
            i += 1
            mark = temp_mark

        return result

    def result_tag_source(self, recommends, source):
        for res in recommends:
            res['source'] = source
        return recommends


class GenerateUser(object):

    def __init__(self, fasttext_filepath):
        self.model = fasttext.load_model(fasttext_filepath)

    def __get_query_vector(self, query):
        vector = self.model.get_sentence_vector(query)
        return vector

    def get_user_embedding(self, querys):
        vectors = []
        for query, weight in querys:
            vector = self.__get_query_vector(query)
            vectors.append(vector)

        user_embedding = np.mean(np.array(vectors), axis=0)
        user_embedding_string = ','.join(map(str, user_embedding.tolist()))
        return user_embedding_string

    def clean_query(self, querys):
        querys = querys.split(u'\u0002')
        querys = [(query.split(':')[0], ':'.join(query.split(':')[1:])) for query in querys if query.count(':') == 1]
        total = sum([float(y) for x, y in querys])
        querys = [(x, (float(y) + 0.0) / total) for x, y in querys]
        return [(x, float(y)) for x, y in querys]


class DropDeduplication(object):

    def __init__(self):
        pass

    def drop_history(self, history, recommend):
        drop = set()
        for h_query in history:
            for ele in recommend:
                r_query = ele['query']
                mark = self.jaccard_v2(h_query, r_query)
                if mark:
                    drop.add(r_query)
        result = [ele for ele in recommend if ele['query'] not in drop]
        return result

    def drop_recommend(self, recommend):

        drop = set()
        length = len(recommend)
        for i in range(length - 1):
            for j in range(i + 1, length):
                word, mark = self.jaccard(recommend[i]['query'], recommend[j]['query'])
                if mark:
                    drop.add(word)

        recommend = [element for element in recommend if element['query'] not in drop and len(element['query']) > 2]
        return recommend

    def jaccard(self, word1_ori, word2_ori):
        mark = False
        word = None
        word1 = set(word1_ori)
        word2 = set(word2_ori)
        if len(word1 - word2) == 0 or len(word2 - word1) == 0:
            mark = True
            word = word2_ori if len(word1) > len(word2) else word1_ori
            return word, mark

        radio = len(word1 & word2) / len(word1 | word2)
        if radio > 0.6:
            mark = True
            word = word2_ori if len(word1) > len(word2) else word1_ori

        return word, mark

    def jaccard_v2(self, word1, word2):
        mark = False

        word1 = set(word1)
        word2 = set(word2)
        if len(word1 - word2) == 0 or len(word2 - word1) == 0:
            mark = True
            return mark

        radio = len(word1 & word2) / len(word1 | word2)
        if radio > 0.6:
            mark = True
        return mark


# 对应的尾号的东西,对应geek
def is_need_geek(geekid, number):
    geekid = int(geekid)
    if geekid % 10 == number:
        return True
    return False


# 加载丢弃的词汇
def load_drop_word(filepath):
    content = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r').strip()
            query = line
            content.add(query)

    return content


# 时间过滤数据
def is_recent_mark(datestring):
    before = datetime.datetime.now() + datetime.timedelta(days=-30)
    today = datetime.datetime.strptime(datestring, '%Y-%m-%d %H:%M:%S.%f')
    if (before - today).days > 0:
        return False
    return True


# 加载城市过滤词表
def load_regex(filepath):
    col = set()
    with open(filepath, 'r') as f:
        for line in f:
            line = line.rstrip('\n').rstrip('\r')
            col.add(line)

    regex = re.compile('|'.join(col))
    return regex


# 挑选高质量的query
def filter_search_query(querys, regex, drop_content, dropObj):
    query_clean = []
    for query in querys:
        if regex.findall(query):
            continue
        if query in drop_content:
            continue

        if '%' in query:
            continue

        query_clean.append({'query': query})

    content = dropObj.drop_recommend(query_clean)
    query_clean = [ele['query'] for ele in content]
    return query_clean


if __name__ == '__main__':
    pass
