//
// Created by admin on 2021/4/20.
//

#ifndef EMBEDDING_RECALL_EMBEDDINGRECALL_H
#define EMBEDDING_RECALL_EMBEDDINGRECALL_H

#include <string>
#include <vector>
#include <unordered_map>

#include <fasttext/fasttext.h>
#include <rocksdb/db.h>
#include <faiss/Index.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>

namespace recall {
    class EmbeddingRecall {

    public:
        EmbeddingRecall(std::string fasttext_filepath, std::string faiss_filepath,
                        std::string rocksdb_filepath);

        void search_query(std::vector<std::string> datas, int k, int nprobe);


    private:
        std::shared_ptr<fasttext::FastText> _fasttext;
        std::shared_ptr<faiss::Index> _faiss;
        std::shared_ptr<rocksdb::DB> _db;

    };
}


#endif //EMBEDDING_RECALL_EMBEDDINGRECALL_H
