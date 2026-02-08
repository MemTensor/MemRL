import types
import unittest


class FakeMOS:
    """A minimal MOS stub for retriever unit tests (no MemOS backend needed)."""

    def __init__(self, *, items=None, hits=None):
        self._items = list(items or [])
        self._hits = list(hits or [])

    def get_all(self, *, user_id: str):
        return {"text_mem": [{"cube_id": "c1", "memories": list(self._items)}]}

    def search(self, *, query: str, user_id: str, top_k: int = 1):
        # Return items without explicit score (TextualMemoryItem)
        return {"text_mem": [{"cube_id": "c1", "memories": list(self._items)[:top_k]}]}

    def search_score(self, *, query: str, user_id: str, top_k: int = 1):
        # Return dict hits with explicit score
        return {"text_mem": [{"cube_id": "c1", "memories": list(self._hits)[:top_k]}]}


class FakeMOSNoSearchScore(FakeMOS):
    """Simulate MemOS versions where MOS.search_score doesn't exist."""

    search_score = None  # attribute exists but is not callable


class TestRetrieversCompat(unittest.TestCase):
    def _make_item(self, mem: str, full_content: str):
        from memos.memories.textual.item import TextualMemoryItem, TextualMemoryMetadata

        md = TextualMemoryMetadata(source="conversation", full_content=full_content)
        return TextualMemoryItem(memory=mem, metadata=md)

    def test_format_memory_result_accepts_textual_item(self):
        from memrl.service import retrievers as r

        itm = self._make_item("q", "FULL")

        out = r._format_memory_result(itm)
        self.assertIn("memory_id", out)
        self.assertEqual(out["content"], "FULL")
        self.assertIn("metadata", out)
        self.assertIn("similarity", out)

    def test_query_retriever_falls_back_when_search_score_missing(self):
        # In some MemOS versions, MOS has no search_score; QueryRetriever must not crash.
        from memrl.service.retrievers import QueryRetriever

        itm = self._make_item("q", "FULL")
        mos = FakeMOSNoSearchScore(items=[itm])
        r = QueryRetriever(mos, user_id="u")

        # threshold>0 and missing score should treat score as 0.0 and thus filter out
        out = r.retrieve("q", k=1, threshold=0.1)
        self.assertEqual(out, [])

        # threshold==0 should allow returning items even without explicit score
        out2 = r.retrieve("q", k=1, threshold=0.0)
        self.assertEqual(len(out2), 1)
        self.assertEqual(out2[0]["content"], "FULL")

    def test_query_retriever_uses_search_score_when_available(self):
        from memrl.service.retrievers import QueryRetriever

        itm = self._make_item("q", "FULL")
        mos = FakeMOS(hits=[{"item": itm, "score": 0.42}])
        r = QueryRetriever(mos, user_id="u")

        out = r.retrieve("q", k=1, threshold=0.4)
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out[0]["similarity"]), 0.42, places=6)

        out2 = r.retrieve("q", k=1, threshold=0.5)
        self.assertEqual(out2, [])

    def test_random_retriever_accepts_textual_items(self):
        from memrl.service.retrievers import RandomRetriever

        itm1 = self._make_item("q1", "FULL1")
        itm2 = self._make_item("q2", "FULL2")
        mos = FakeMOS(items=[itm1, itm2])
        r = RandomRetriever(mos, user_id="u")

        out = r.retrieve("ignored", k=1, threshold=0.0)
        self.assertEqual(len(out), 1)
        self.assertIn(out[0]["content"], {"FULL1", "FULL2"})


if __name__ == "__main__":
    unittest.main()

