"""Memory Graph with PPR (Personalized PageRank) diffusion.

从 hindsight entity/relation 标签构建轻量级有向图，
支持 Personalized PageRank 多跳扩散用于联想记忆扩展。

用法:
    mg = MemoryGraph()
    mg.build_from_hindsight()
    scores = mg.ppr(seed_nodes=["entity|object:ichimoku_fix"])
    # → {"entity|object:ichimoku_fix": 0.45, "entity|object:holding_analysis": 0.28, ...}
"""

import json
import logging
import re
import time
from collections import defaultdict
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

API_BASE = "http://127.0.0.1:9177/v1/default/banks/hermes"


class MemoryGraph:
    """轻量级内存图，从 hindsight entity/relation 标签构建邻接表。"""

    def __init__(self):
        # 邻接表: {node: [(neighbor, weight, relation_type), ...]}
        self.graph: dict[str, list[tuple[str, float, str]]] = {}
        # 节点→该节点的记忆文本(最多3条用于上下文)
        self.node_context: dict[str, list[str]] = defaultdict(list)
        # 节点→类型
        self.node_types: dict[str, str] = {}
        # 构建时间戳
        self._built_at: float = 0
        # 新鲜度阈值(秒): 30分钟
        self._ttl: float = 1800

    def build_from_hindsight(self, force: bool = False) -> int:
        """从 hindsight API 拉取所有 entity/relation 标签构建图。

        Args:
            force: 如果 True, 跳过 TTL 检查强制重建

        Returns:
            图的节点数
        """
        now = time.time()
        if not force and self._built_at and (now - self._built_at) < self._ttl:
            logger.debug("MemoryGraph: TTL valid, using cached graph (%d nodes)", len(self.graph))
            return len(self.graph)

        logger.debug("MemoryGraph: building from hindsight...")
        self.graph.clear()
        self.node_context.clear()
        self.node_types.clear()

        # 1. 拉取所有记忆
        memories = self._fetch_all_memories()
        logger.debug("MemoryGraph: fetched %d memories", len(memories))

        # 2. 扫描 entity/relation 标签建图
        entity_re = re.compile(r"^entity\|(\w+):(.+)$")
        relation_re = re.compile(r"^relation\|(\w+):(.+)$")

        for mem in memories:
            text = mem.get("text", "") or mem.get("content", "") or ""
            tags = mem.get("tags", []) or []

            current_entity = None
            current_etype = None
            relations = []

            for tag in tags:
                tag_s = str(tag)
                m = entity_re.match(tag_s)
                if m:
                    current_entity = f"entity|{m.group(1)}:{m.group(2)}"
                    current_etype = m.group(1)
                    continue
                m = relation_re.match(tag_s)
                if m:
                    relations.append((m.group(1), m.group(2)))

            if current_entity:
                # 记录节点类型
                self.node_types[current_entity] = current_etype or "unknown"

                # 记录节点关联的上下文(最多3条)
                if text and len(self.node_context[current_entity]) < 3:
                    self.node_context[current_entity].append(text[:200])

                # 建边: entity → relation target
                for prefix, target in relations:
                    target_node = f"entity|{prefix}:{target}"
                    # 权重: 相同 prefix 累加
                    self._add_edge(current_entity, target_node, 1.0, prefix)

        self._built_at = time.time()
        logger.debug("MemoryGraph: built %d nodes, %d edges",
                     len(self.graph), sum(len(v) for v in self.graph.values()))
        return len(self.graph)

    def ppr(self, seed_nodes: list[str],
            alpha: float = 0.85,
            max_iter: int = 50,
            tol: float = 1e-6) -> dict[str, float]:
        """Personalized PageRank 图扩散。

        Args:
            seed_nodes: 种子节点列表(如 ["entity|object:ichimoku_fix"])
            alpha: 随机游走中跳转到种子节点的概率(0.85=经典值)
            max_iter: 最大迭代次数
            tol: 收敛容差

        Returns:
            {node: ppr_score} 按分数降序排列
        """
        if not seed_nodes:
            return {}

        # 归一化种子分数
        n = len(seed_nodes)
        seed_score = 1.0 / n

        # 初始化: 种子节点均匀分布
        scores: dict[str, float] = {}
        for node in seed_nodes:
            if node in self.graph or node in self.node_context:
                scores[node] = seed_score

        if not scores:
            logger.debug("PPR: no seed nodes found in graph")
            return {}

        # 其余节点初始化为0
        for node in self.graph:
            if node not in scores:
                scores[node] = 0.0

        # 迭代 PPR
        for iteration in range(max_iter):
            max_delta = 0.0
            new_scores: dict[str, float] = {}

            # 计算每个节点的 PPR 值
            for node in scores:
                # 随机游走到种子: alpha * seed_score
                restart = alpha * (seed_score if node in seed_nodes else 0.0)

                # 从邻接节点传播过来
                propagate = 0.0
                neighbors = self.graph.get(node, [])
                for neighbor, weight, _ in neighbors:
                    # 获取邻居的度(出度)
                    out_degree = len(self.graph.get(neighbor, []))
                    if out_degree > 0:
                        propagate += (1.0 - alpha) * scores.get(neighbor, 0.0) * weight / out_degree

                new_scores[node] = restart + propagate

                # 计算变化量
                delta = abs(new_scores[node] - scores.get(node, 0.0))
                if delta > max_delta:
                    max_delta = delta

            scores = new_scores

            if max_delta < tol:
                logger.debug("PPR: converged at iteration %d, max_delta=%e", iteration + 1, max_delta)
                break

        # 归一化
        total = sum(scores.values())
        if total > 0:
            for node in scores:
                scores[node] /= total

        # 按分数降序排列
        return dict(sorted(scores.items(), key=lambda x: -x[1]))

    def get_expanded_nodes(self, seed_nodes: list[str],
                           top_k: int = 5,
                           min_score: float = 0.01) -> list[tuple[str, float]]:
        """从种子节点扩散，返回前 top_k 个关联节点。

        Args:
            seed_nodes: 种子节点
            top_k: 返回最多 top_k 个
            min_score: 最低分数阈值

        Returns:
            [(node, score), ...] 按分数降序
        """
        if not self.graph or not self._built_at:
            self.build_from_hindsight()

        scores = self.ppr(seed_nodes)
        # 排除种子节点本身
        seed_set = set(seed_nodes)
        results = [(n, s) for n, s in scores.items()
                   if n not in seed_set and s >= min_score]
        return results[:top_k]

    def _add_edge(self, src: str, dst: str, weight: float = 1.0,
                  rel_type: str = "unknown"):
        """添加有向边 src → dst。"""
        if src not in self.graph:
            self.graph[src] = []
        # 检查是否已存在同类型边
        for i, (existing_dst, existing_weight, existing_type) in enumerate(self.graph[src]):
            if existing_dst == dst and existing_type == rel_type:
                self.graph[src][i] = (dst, existing_weight + weight, rel_type)
                return
        self.graph[src].append((dst, weight, rel_type))

    def _fetch_all_memories(self, limit: int = 500) -> list[dict]:
        """拉取所有 hindsight 记忆。"""
        url = f"{API_BASE}/memories/list?limit={limit}"
        try:
            req = Request(url)
            resp = urlopen(req, timeout=30)
            data = json.loads(resp.read().decode())
            if isinstance(data, list):
                return data
            return data.get("results", data.get("items", data.get("memories", [])))
        except Exception as e:
            logger.warning("MemoryGraph: fetch failed: %s", e)
            return []

    def stats(self) -> dict:
        """返回图统计信息。"""
        return {
            "nodes": len(self.graph),
            "edges": sum(len(v) for v in self.graph.values()),
            "node_types": dict(self.node_types),
            "built_at": self._built_at,
            "age_seconds": time.time() - self._built_at if self._built_at else -1,
        }
