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
}



