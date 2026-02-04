import os
import json
import numpy as np
import faiss  # 需要安装 faiss-cpu 或 faiss-gpu
import aiohttp # 需要安装 aiohttp
import asyncio
from typing import List, Dict, Any, Optional

class AsyncFaissRAG:
    """
    一个使用 FAISS 和 JSON 文件存储的、支持异步操作的简单RAG系统。
    - FAISS 用于存储和搜索文档键 (key) 的嵌入向量。
    - JSON 用于存储文档的 ID、键 (key)、值 (value) 和元数据。
    - 所有 I/O 操作（网络、文件）均为异步。
    """
    
    def __init__(
        self, 
        json_path: str = "rag_data.json", 
        faiss_index_path: str = "rag_index.faiss",
        api_url: str = "http://localhost:6007/v1",
        model_name: str = "qwen3-emb"
    ):
        self.api_url = api_url
        self.model_name = model_name
        self.json_path = json_path
        self.faiss_index_path = faiss_index_path
        # --- [关键改动 1] 初始化一个异步锁 ---
        self.lock = asyncio.Lock()
        self.documents: Dict[int, Dict[str, Any]] = {}
        self.index: Optional[faiss.IndexIDMap] = None
        self.dimension: Optional[int] = None

    @classmethod
    async def create(cls, *args, **kwargs) -> 'AsyncFaissRAG':
        """异步工厂方法，用于创建和初始化实例。"""
        instance = cls(*args, **kwargs)
        await instance._load()
        return instance

    async def _load(self):
        """从文件异步加载文档和FAISS索引"""
        # 在线程中执行同步的文件读取操作，避免阻塞事件循环
        if os.path.exists(self.json_path):
            def sync_load_json():
                with open(self.json_path, 'r', encoding='utf-8') as f:
                    return {int(k): v for k, v in json.load(f).items()}
            self.documents = await asyncio.to_thread(sync_load_json)
            print(f"从 {self.json_path} 加载了 {len(self.documents)} 个文档。")

        if os.path.exists(self.faiss_index_path):
            def sync_load_faiss():
                return faiss.read_index(self.faiss_index_path)
            self.index = await asyncio.to_thread(sync_load_faiss)
            self.dimension = self.index.d
            print(f"从 {self.faiss_index_path} 加载了 FAISS 索引，维度: {self.dimension}。")

    async def _save(self):
        """将文档和FAISS索引异步保存到文件。这个方法本身不加锁，由调用它的方法负责加锁。"""
        # --- [关键改动 2] 创建 self.documents 的浅拷贝用于保存 ---
        # 这样即使在 dump 过程中有其他地方（理论上不应该，因为有锁）修改了 self.documents，
        # 也不会影响当前正在保存的数据。这是更健壮的做法。
        documents_to_save = self.documents.copy()

        def sync_save_json():
            with open(self.json_path, 'w', encoding='utf-8') as f:
                json.dump(documents_to_save, f, ensure_ascii=False, indent=4)
        
        def sync_save_faiss():
            if self.index:
                faiss.write_index(self.index, self.faiss_index_path)

        await asyncio.gather(
            asyncio.to_thread(sync_save_json),
            asyncio.to_thread(sync_save_faiss)
        )
        print("数据和索引已异步保存。")


    async def get_embedding(self, text: str) -> Optional[List[float]]:
        """使用API异步获取文本嵌入"""
        payload = {
            "model": self.model_name,
            "input": text
        }
        headers = {"Content-Type": "application/json"}
        
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(f"{self.api_url}/embeddings", json=payload, headers=headers) as response:
                    response.raise_for_status()
                    result = await response.json()
                    embedding = result["data"][0]["embedding"]
                    
                    embedding_np = np.array(embedding, dtype='float32')
                    faiss.normalize_L2(embedding_np.reshape(1, -1))
                    return embedding_np.flatten().tolist()
        except Exception as e:
            print(f"异步获取嵌入失败: {e}")
            return None

    async def add_document(self, key: str, value: str, metadata: Dict[str, Any] = None) -> Optional[int]:
        """异步添加一个键值对文档到RAG系统。"""
        if metadata is None:
            metadata = {}
        
        embedding = await self.get_embedding(key)
        if embedding is None:
            print("无法添加文档，因为获取嵌入失败。")
            return None
        
        embedding_np = np.array([embedding], dtype='float32')
        
        # --- [关键改动 3] 在修改共享状态之前获取锁 ---
        async with self.lock:
            if self.index is None:
                self.dimension = embedding_np.shape[1]
                index_flat = faiss.IndexFlatIP(self.dimension)
                self.index = faiss.IndexIDMap(index_flat)
                print(f"已初始化 FAISS 索引，维度: {self.dimension}")

            if embedding_np.shape[1] != self.dimension:
                print(f"错误: 嵌入维度 ({embedding_np.shape[1]}) 与索引维度 ({self.dimension}) 不匹配。")
                return None

            doc_id = max(self.documents.keys()) + 1 if self.documents else 1
            
            # 修改 index 和 documents
            self.index.add_with_ids(embedding_np, np.array([doc_id]))
            self.documents[doc_id] = { "key": key, "value": value, "metadata": metadata }
            
            # 调用保存
            await self._save()
        
        return doc_id

    async def search(self, query: str, top_k: int = 5) -> List[Dict[str, Any]]:
        """异步地根据查询与存储的'key'进行相似度搜索。"""
        async with self.lock:
            if self.index is None or not self.documents:
                print("系统中没有文档，无法搜索。")
                return []
            
        query_embedding = await self.get_embedding(query)
        if query_embedding is None:
            return []
        
        query_embedding_np = np.array([query_embedding], dtype='float32')
        
        # FAISS search本身是CPU密集型，同步执行即可
        similarities, doc_ids = self.index.search(query_embedding_np, top_k)
        
        results = []
        # 再次获取锁以进行安全的搜索和数据读取
        async with self.lock:
            for i in range(len(doc_ids[0])):
                doc_id = doc_ids[0][i]
                if doc_id in self.documents:
                    doc = self.documents[doc_id]
                    results.append({
                        "id": int(doc_id),
                        "key": doc["key"],
                        "value": doc["value"],
                        "metadata": doc["metadata"],
                        "similarity": float(similarities[0][i])
                    })
        
        return results

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        """
        异步计算两个文本之间的余弦相似度。
        原理：由于 get_embedding 已经做了 L2 归一化，点积即为余弦相似度。
        """
        # 1. 并发获取两个文本的嵌入向量，提高效率
        emb1_task = self.get_embedding(text1)
        emb2_task = self.get_embedding(text2)
        
        # 等待两个任务完成
        emb1, emb2 = await asyncio.gather(emb1_task, emb2_task)

        # 2. 检查嵌入是否获取成功
        if emb1 is None or emb2 is None:
            print(f"计算相似度失败：无法获取嵌入 (Text1: {'成功' if emb1 else '失败'}, Text2: {'成功' if emb2 else '失败'})")
            return 0.0

        # 3. 计算点积
        # 注意：get_embedding 返回的是 list，建议转为 numpy array 计算更高效
        vec1 = np.array(emb1, dtype='float32')
        vec2 = np.array(emb2, dtype='float32')

        # 计算点积
        similarity = np.dot(vec1, vec2)

        # 确保返回的是 Python float 而不是 numpy float
        return float(similarity)
    def get_document(self, doc_id: int) -> Optional[Dict[str, Any]]:
        """通过ID获取单个文档（同步操作，因为是内存访问）。"""
        return self.documents.get(doc_id)

    def list_documents(self, limit: int = 100) -> List[Dict[str, Any]]:
        """列出所有文档（同步操作）。"""
        return list(self.documents.values())[:limit]

    async def delete_document(self, doc_id: int) -> bool:
        """异步地通过ID删除文档"""
        # --- [关键改动 4] 在修改共享状态之前获取锁 ---
        doc_id = int(doc_id)
        async with self.lock:
            if doc_id not in self.documents:
                print(f"文档ID {doc_id} 不存在。")
                return False
            
            try:
                # FAISS remove_ids是CPU密集型，同步执行
                self.index.remove_ids(np.array([doc_id]))
                del self.documents[doc_id]
                
                await self._save()
                print(f"文档ID {doc_id} 已成功删除。")
                return True
            except Exception as e:
                print(f"删除文档ID {doc_id} 失败: {e}")
                return False

# async def main():
#     """示例使用（异步版本）"""
    # # 清理旧文件以便重新演示
    # if os.path.exists("rag_data.json"): os.remove("rag_data.json")
    # if os.path.exists("rag_index.faiss"): os.remove("rag_index.faiss")
        
    # # 使用异步工厂方法创建实例
    # rag = await AsyncFaissRAG.create()
    
#     print("\n=== 异步 FAISS + JSON RAG系统演示 ===")
    
#     sample_docs = [
#         {"key": "人工智能的定义", "value": "人工智能（AI）是计算机科学的一个前沿分支..."},
#         {"key": "机器学习是什么", "value": "机器学习是人工智能的一个核心子领域..."},
#         {"key": "深度学习的应用领域", "value": "深度学习通过构建深层神经网络..."}
#     ]
    
#     print("\n异步添加文档...")
    # add_tasks = [rag.add_document(doc["key"], doc["value"]) for doc in sample_docs]
    # doc_ids = await asyncio.gather(*add_tasks)
#     for doc, doc_id in zip(sample_docs, doc_ids):
#         if doc_id:
#             print(f"文档已添加，ID: {doc_id}, Key: '{doc['key']}'")

#     queries = ["介绍一下机器学习", "深度学习用在什么地方？", "AI的全称是什么"]
    
#     print("\n--- 异步搜索测试 ---")
#     async def run_search(query):
#         print(f"\n[查询]: {query}")
#         results = await rag.search(query, top_k=2)
        # if not results:
        #     print("  未找到相关结果。")
        #     return
        # for i, result in enumerate(results, 1):
        #     print(f"  {i}. 相似度: {result['similarity']:.4f} (与Key: '{result['key']}')")
        #     print(f"     返回内容(Value): {result['value'][:100]}...")

#     search_tasks = [run_search(q) for q in queries]
#     await asyncio.gather(*search_tasks)

#     print("\n--- 异步删除测试 ---")
#     doc_to_delete_id = 2
#     print(f"尝试删除文档ID: {doc_to_delete_id}...")
#     await rag.delete_document(doc_to_delete_id)
    
#     print(f"\n再次搜索 '介绍一下机器学习' (之前最相关的文档已被删除):")
#     await run_search("介绍一下机器学习")

#     print("\n演示完成！")

# if __name__ == "__main__":
#     # 确保你的嵌入API服务正在运行
#     # 运行主异步函数
#     try:
#         asyncio.run(main())
#     except KeyboardInterrupt:
#         print("\n程序被用户中断。")