//
// Created by admin on 2021/4/20.
//

#include "EmbeddingRecall.h"

namespace recall {
    EmbeddingRecall::EmbeddingRecall(std::string fasttext_filepath, std::string faiss_filepath,
                                     std::string rocksdb_filepath) {

        this->_fasttext.reset(new fasttext::FastText());
        this->_fasttext->loadModel(fasttext_filepath);

        this->_faiss.reset((faiss::IndexIVFPQ *) faiss::read_index(faiss_filepath.c_str()));

        rocksdb::Options options;
        options.create_if_missing = true;
        rocksdb::DB *db = nullptr;
        rocksdb::DB::Open(options, rocksdb_filepath, &db);
        this->_db.reset(db);

    }

    bool EmbeddingRecall::query_vector(const std::vector<std::string> &querys, float *datas, int column) {

        for (int i = 0; i < querys.size(); i++) {
            fasttext::Vector vector(100);
            std::stringstream ss;
            ss.clear();
            ss << querys[i];
            this->_fasttext->getSentenceVector(ss, vector);
            for (int j = 0; j < vector.size(); j++) {
                datas[i * column + j] = vector[j];
            }
        }
        return true;
    }

    bool EmbeddingRecall::search_neighbor(int count, float *data, int k, float *distance, long long *index) {
        this->_faiss->search(count, data, k, distance, index);
        return false;
    }

    bool EmbeddingRecall::forward_index(int count, int k, float *distance, long long int *index) {

        float dis = 0.0f;
        long long int idx = 0;
        for (int i = 0; i < count; i++) {
            for (int j = 0; j < k; j++) {

                dis = *(distance + i * k + j);
                idx = *(index + i * k + j);
            }
        }

        return true;
    }


}



