from qdrant_client import models
from src.db import get_client, COLLECTION_NAME
from src.embedders import VisualEngine

class CreativeSearch:
    def __init__(self):
        self.client = get_client()
        self.engine = VisualEngine()

    def search(self, text_query, hard_filters=None, soft_filters=None, limit=10):
        """
        text_query: vibe string
        hard_filters: must match
        soft_filters: should match (boost)
        """
        print(f"🔎 Vibe: '{text_query}' | Hard: {hard_filters} | Soft: {soft_filters}")

        # 1. vectorize 
        vector = self.engine.get_text_vector(text_query)

        # 2. build hard constraints
        qdrant_filter = None
        if hard_filters:
            must_conditions = [
                models.FieldCondition(key=k, match=models.MatchValue(value=v))
                for k, v in hard_filters.items()
            ]
            qdrant_filter = models.Filter(must=must_conditions)

        # 3. execute search (fetch extra for reranking)
        # 3x limit to catch "soft matches"
        pool_size = limit * 3
        hits = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=vector,
            query_filter=qdrant_filter,
            limit=pool_size,
            with_payload=True
        ).points

        # 4. apply soft filters (score boost)
        # manipulate score in python
        if soft_filters:
            hits = self._apply_boosting(hits, soft_filters)

        # 5. re-sort and slice
        # sort by boosted score (desc)
        hits.sort(key=lambda x: x.score, reverse=True)
        return hits[:limit]

    def _apply_boosting(self, hits, soft_filters):
        """
        boosts score if matches soft filters.
        strategy: +0.15 per match.
        """
        BOOST_STRENGTH = 0.15 
        
        for hit in hits:
            # check every soft filter key/value
            for key, target_val in soft_filters.items():
                actual_val = hit.payload.get(key)
                
                # check for exact match (case insensitive recommended for prod)
                if actual_val == target_val:
                    hit.score += BOOST_STRENGTH
                    # mark as boosted (optional debug)
                    if "boosted" not in hit.payload:
                        hit.payload["_debug_boost"] = []
                    hit.payload["_debug_boost"].append(key)
        
        return hits