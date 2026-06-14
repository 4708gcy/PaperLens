"""验证 ES 9.3.1 的 dense_vector + KNN 接口（用 4 维小向量快速验证 API 语法）"""
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk

es = Elasticsearch(hosts=["http://localhost:9200"])
TEST_IDX = "paperlens_knn_smoke_test"


def main():
    # 清理旧索引
    if es.indices.exists(index=TEST_IDX):
        es.indices.delete(index=TEST_IDX)

    # 建 dense_vector 索引（与项目实际 mapping 一致，只是维度改 4）
    mapping = {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {"properties": {
            "content": {"type": "text", "analyzer": "ik_max_word"},
            "vec": {
                "type": "dense_vector",
                "dims": 4,
                "index": True,
                "similarity": "cosine",
                "index_options": {"type": "hnsw", "m": 16, "ef_construction": 256}
            }
        }}
    }
    es.indices.create(index=TEST_IDX, body=mapping)
    print("[1/4] 索引创建成功（dense_vector + hnsw + cosine）")

    # bulk 写入
    docs = [
        {"content": "attention is all you need", "vec": [1.0, 0.0, 0.0, 0.0]},
        {"content": "bert pretraining", "vec": [0.0, 1.0, 0.0, 0.0]},
        {"content": "gpt generation", "vec": [0.0, 0.0, 1.0, 0.0]},
        {"content": "another attention paper", "vec": [0.9, 0.1, 0.0, 0.0]},
    ]
    actions = [{"_index": TEST_IDX, "_source": d} for d in docs]
    success, _ = bulk(es, actions)
    es.indices.refresh(index=TEST_IDX)
    print(f"[2/4] bulk 写入 {success} 条")

    # KNN 查询（查与 [1,0,0,0] 最相似的）
    resp = es.search(index=TEST_IDX, body={
        "knn": {"field": "vec", "query_vector": [1.0, 0.0, 0.0, 0.0], "k": 2, "num_candidates": 4},
        "size": 2,
        "_source": ["content"]
    })
    hits = resp["hits"]["hits"]
    print(f"[3/4] KNN 查询返回 {len(hits)} 条:")
    for h in hits:
        print(f"        score={h['_score']:.4f} content={h['_source']['content']}")

    # BM25 查询
    resp2 = es.search(index=TEST_IDX, body={
        "query": {"match": {"content": "attention"}},
        "size": 2
    })
    print(f"[4/4] BM25 查询返回 {len(resp2['hits']['hits'])} 条 attention 相关文档")

    # 清理
    es.indices.delete(index=TEST_IDX)
    print("\n[ALL OK] ES 9.3.1 dense_vector KNN + BM25 + IK 全部验证通过")


if __name__ == "__main__":
    main()
