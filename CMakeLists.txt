CMAKE_MINIMUM_REQUIRED(VERSION 3.17)
PROJECT(EmbeddingRecall)

# 头文件设置
INCLUDE_DIRECTORIES(
        "~/workspace/code/cpp/incubator-brpc/output/include"
        "/usr/local/Cellar/fasttext/0.9.2/include"
        "/usr/local/Cellar/faiss/1.6.3/include")
INCLUDE_DIRECTORIES("src/embedding_recall")
# 其他目录的文件
## embedding_recall的文件
AUX_SOURCE_DIRECTORY(src/embedding_recall embedding_recall_code)
ADD_SUBDIRECTORY(src/embedding_recall)
## 最后的文件
ADD_EXECUTABLE(${PROJECT_NAME} main.cpp ${embedding_recall_code})

