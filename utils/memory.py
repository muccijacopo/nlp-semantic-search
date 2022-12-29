class MemoryManager:
    _entities = {}

    def save_entity(self, key: str, value: str):
        self._entities[key] = value
        return self._entities

    def get_entity(self, key):
        return self._entities.get(key) or None


memory_manager = MemoryManager()
