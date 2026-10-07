"""Real service paths with only external MemOS import types substituted."""
import copy
import importlib
import sys
import threading
from types import ModuleType, SimpleNamespace
from unittest.mock import patch


def load_service():
    modules = {}
    names = {
        "memos.configs.mem_os": ["MOSConfig"],
        "memos.configs.mem_cube": ["GeneralMemCubeConfig"],
        "memos.mem_os.main": ["MOS"],
        "memos.mem_cube.general": ["GeneralMemCube"],
        "memos.memories.textual.item": ["TextualMemoryItem", "TextualMemoryMetadata"],
        "memos.utils": [],
    }
    for name, types in names.items():
        parts = name.split(".")
        for end in range(1, len(parts) + 1):
            path = ".".join(parts[:end])
            if path not in modules:
                modules[path] = ModuleType(path)
                modules[path].__path__ = []
        for type_name in types:
            setattr(modules[name], type_name, type(type_name, (), {}))
    with patch.dict(sys.modules, modules):
        return importlib.import_module("memrl.service.memory_service")


service_module = load_service()


def make_service(items, vectors):
    module = service_module
    service = module.MemoryService.__new__(module.MemoryService)
    service.rl_config = module.RLConfig(epsilon=0, topk=1)
    service.dict_memory = {}
    for memory_id, item in items.items():
        service.dict_memory.setdefault(item.memory, []).append(memory_id)
    service.query_embeddings = vectors.copy()
    service.embedding_provider = SimpleNamespace(
        embed=lambda texts: [vectors[text] for text in texts]
    )
    service._mem_cache = copy.deepcopy(items)
    service._mem_cache_max_size = 20
    service._q_cache = {}
    service._q_cache_max_size = 20
    service.mos = SimpleNamespace(
        get=lambda **kwargs: copy.deepcopy(items[kwargs["memory_id"]])
    )
    service.default_cube_id = "cube"
    service.user_id = "synthetic"
    service._db_gate = threading.BoundedSemaphore(1)
    service.weight_sim = 1.0
    service.weight_q = 0.0
    service.use_z_score_normalization = False
    service.dedup_by_task_id = False
    return service
