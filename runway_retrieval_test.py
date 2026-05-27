import os
import json
import math
import pandas as pd
import torch
import torch.nn as nn

try:
    from safetensors.torch import load_file as safe_load_file
    SAFETENSORS_AVAILABLE = True
except ImportError:
    SAFETENSORS_AVAILABLE = False

from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim


# =========================================================
# 路径设置：所有文件均在当前脚本所在根目录下
# =========================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join("output", "GeometryBridge-sbert")
QUERY_JSON_PATH = os.path.join(BASE_DIR, "runway-queries.json")
DOCUMENT_JSON_PATH = os.path.join(BASE_DIR, "runway-document.json")
CSV_PATH = os.path.join(BASE_DIR, "test.csv")

TOP_K = 10
BATCH_SIZE = 64
USE_CPU = False


# =========================================================
# GeometryBridge：必须与训练代码中的自定义模块保持一致
# =========================================================

class GeometryBridge(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        rank_size: int = 192,
        alpha: float = 0.0,
        use_layernorm: bool = True,
    ) -> None:
        super().__init__()

        self.hidden_size = int(hidden_size)
        self.rank_size = int(rank_size)
        self.alpha = float(alpha)
        self.use_layernorm = bool(use_layernorm)

        self.down_proj = nn.Linear(self.hidden_size, self.rank_size)
        self.up_proj = nn.Linear(self.rank_size, self.hidden_size)
        self.activation = nn.GELU()
        self.layernorm = nn.LayerNorm(self.hidden_size) if self.use_layernorm else nn.Identity()

    def forward(self, features):
        token_embeddings = features.get("token_embeddings")

        if token_embeddings is None:
            return features

        bridge = self.up_proj(
            self.activation(
                self.down_proj(token_embeddings)
            )
        )

        fused = self.alpha * token_embeddings + (1.0 - self.alpha) * bridge
        features["token_embeddings"] = self.layernorm(fused)

        return features

    def get_config_dict(self):
        return {
            "hidden_size": self.hidden_size,
            "rank_size": self.rank_size,
            "alpha": self.alpha,
            "use_layernorm": self.use_layernorm,
        }

    def save(self, output_path: str, *args, **kwargs) -> None:
        os.makedirs(output_path, exist_ok=True)

        with open(os.path.join(output_path, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.get_config_dict(), f, ensure_ascii=False, indent=2)

        torch.save(
            self.state_dict(),
            os.path.join(output_path, "pytorch_model.bin")
        )

    @staticmethod
    def load(input_path: str, *args, **kwargs):
        config_path = os.path.join(input_path, "config.json")

        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)

        module = GeometryBridge(
            hidden_size=int(cfg.get("hidden_size", 768)),
            rank_size=int(cfg.get("rank_size", 192)),
            alpha=float(cfg.get("alpha", 0.0)),
            use_layernorm=bool(cfg.get("use_layernorm", True)),
        )

        safetensors_path = os.path.join(input_path, "model.safetensors")
        bin_path = os.path.join(input_path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            if not SAFETENSORS_AVAILABLE:
                raise ImportError(
                    "检测到 model.safetensors，但当前环境没有安装 safetensors。"
                    "请运行：pip install safetensors"
                )
            state = safe_load_file(safetensors_path, device="cpu")
            module.load_state_dict(state, strict=False)

        elif os.path.exists(bin_path):
            state = torch.load(bin_path, map_location="cpu")
            module.load_state_dict(state, strict=False)

        else:
            print(f"Warning: GeometryBridge 权重文件不存在: {input_path}")

        return module


def load_json(path):
    with open(path, "r", encoding="utf-8-sig") as f:
        data = json.load(f)

    return {str(k): str(v) for k, v in data.items()}


def load_label_csv(csv_path):
    """
    读取 test.csv。

    test.csv 格式：
    query_id document_id label_score
    """

    df = pd.read_csv(
        csv_path,
        sep=r"\s+",
        header=None,
        names=["query_id", "document_id", "label_score"],
        engine="python"
    )

    df["query_id"] = df["query_id"].astype(str).str.strip()
    df["document_id"] = df["document_id"].astype(str).str.strip()

    df["label_score"] = pd.to_numeric(
        df["label_score"],
        errors="coerce"
    ).fillna(0).astype(int)

    test_query_ids = set(df["query_id"].tolist())

    label_dict = {
        (row["query_id"], row["document_id"]): int(row["label_score"])
        for _, row in df.iterrows()
    }

    qrels = {}

    for _, row in df.iterrows():
        qid = row["query_id"]
        doc_id = row["document_id"]
        rel = int(row["label_score"])

        if qid not in qrels:
            qrels[qid] = {}

        qrels[qid][doc_id] = rel

    return label_dict, qrels, test_query_ids


def check_file_exists():
    paths = [
        MODEL_PATH,
        QUERY_JSON_PATH,
        DOCUMENT_JSON_PATH,
        CSV_PATH,
    ]

    for path in paths:
        if not os.path.exists(path):
            raise FileNotFoundError(f"文件或目录不存在: {path}")


def calculate_mrr_at_k(top_docs, rel_dict, k=10):
    """
    MRR@K：
    Top-K 中第一个 label_score > 0 的文档的倒数排名。
    """

    for rank, doc_id in enumerate(top_docs[:k], start=1):
        if rel_dict.get(doc_id, 0) > 0:
            return 1.0 / rank

    return 0.0


def calculate_ndcg_at_k(top_docs, rel_dict, k=10):
    """
    NDCG@K：
    使用 test.csv 中的 label_score 作为相关性等级。
    """

    dcg = 0.0

    for rank, doc_id in enumerate(top_docs[:k], start=1):
        rel = rel_dict.get(doc_id, 0)

        if rel <= 0:
            continue

        gain = (2 ** rel) - 1
        dcg += gain / math.log2(rank + 1)

    ideal_rels = sorted(
        [rel for rel in rel_dict.values() if rel > 0],
        reverse=True
    )

    idcg = 0.0

    for rank, rel in enumerate(ideal_rels[:k], start=1):
        gain = (2 ** rel) - 1
        idcg += gain / math.log2(rank + 1)

    if idcg == 0:
        return 0.0

    return dcg / idcg


def calculate_precision_at_k(top_docs, rel_dict, k=10):
    """
    Precision@K：
    Top-K 中 label_score > 0 的文档数量 / K。
    """

    if k <= 0:
        return 0.0

    relevant_count = 0

    for doc_id in top_docs[:k]:
        if rel_dict.get(doc_id, 0) > 0:
            relevant_count += 1

    return relevant_count / k


def main():
    check_file_exists()

    device = "cuda" if torch.cuda.is_available() and not USE_CPU else "cpu"
    print(f"Using device: {device}")

    queries = load_json(QUERY_JSON_PATH)
    documents = load_json(DOCUMENT_JSON_PATH)
    label_dict, qrels, test_query_ids = load_label_csv(CSV_PATH)

    print(f"Loaded queries: {len(queries)}")
    print(f"Loaded documents: {len(documents)}")
    print(f"Loaded label pairs: {len(label_dict)}")
    print(f"Queries appearing in test.csv: {len(test_query_ids)}")

    # 只保留 test.csv 中出现过的 query
    filtered_query_ids = [
        qid for qid in queries.keys()
        if qid in test_query_ids
    ]

    if not filtered_query_ids:
        raise ValueError("没有找到任何出现在 test.csv 中的 query_id，请检查 query_id 是否一致。")

    filtered_query_texts = [
        queries[qid]
        for qid in filtered_query_ids
    ]

    print(f"Evaluated queries: {len(filtered_query_ids)}")

    model = SentenceTransformer(
        MODEL_PATH,
        device=device
    )

    doc_ids = list(documents.keys())
    doc_texts = [documents[doc_id] for doc_id in doc_ids]

    print("Encoding documents...")

    doc_embeddings = model.encode(
        doc_texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    print("Encoding queries...")

    query_embeddings = model.encode(
        filtered_query_texts,
        batch_size=BATCH_SIZE,
        convert_to_tensor=True,
        normalize_embeddings=True,
        show_progress_bar=True
    )

    mrr_list = []
    ndcg_list = []
    precision_list = []

    print("Calculating MRR@10, NDCG@10, Precision@10...")

    for q_idx, qid in enumerate(filtered_query_ids):
        q_emb = query_embeddings[q_idx]

        scores = cos_sim(q_emb, doc_embeddings)[0]

        top_k = min(TOP_K, len(doc_ids))
        top_results = torch.topk(scores, k=top_k)

        top_doc_ids = [
            doc_ids[idx]
            for idx in top_results.indices.tolist()
        ]

        rel_dict = qrels.get(qid, {})

        mrr = calculate_mrr_at_k(top_doc_ids, rel_dict, k=TOP_K)
        ndcg = calculate_ndcg_at_k(top_doc_ids, rel_dict, k=TOP_K)
        precision = calculate_precision_at_k(top_doc_ids, rel_dict, k=TOP_K)

        mrr_list.append(mrr)
        ndcg_list.append(ndcg)
        precision_list.append(precision)

    avg_mrr = sum(mrr_list) / len(mrr_list) if mrr_list else 0.0
    avg_ndcg = sum(ndcg_list) / len(ndcg_list) if ndcg_list else 0.0
    avg_precision = sum(precision_list) / len(precision_list) if precision_list else 0.0

    print("\n" + "=" * 60)
    print("Final Retrieval Evaluation Results")
    print("=" * 60)
    print(f"Evaluated queries: {len(filtered_query_ids)}")
    print(f"Top-K: {TOP_K}")
    print(f"MRR@{TOP_K}: {avg_mrr:.6f}")
    print(f"NDCG@{TOP_K}: {avg_ndcg:.6f}")
    print(f"Recall@{TOP_K}: {avg_precision:.6f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
