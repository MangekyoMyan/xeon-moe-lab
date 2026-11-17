class RingQueue:
    """Fixed-capacity FIFO queue of signed 64-bit integers."""

    def __init__(self, capacity: int) -> None:
        if not (1 <= capacity <= 1024):
            raise ValueError("capacity must be between 1 and 1024")
        self._buf = [0] * capacity
        self._capacity = capacity
        self._head = 0   # index of the oldest element
        self._count = 0  # number of stored elements

    def size(self) -> int:
        return self._count

    def try_push(self, value: int) -> bool:
        if self._count == self._capacity:
            return False
        tail = (self._head + self._count) % self._capacity
        self._buf[tail] = value
        self._count += 1
        return True

    def try_pop(self):
        if self._count == 0:
            return None
        value = self._buf[self._head]
        self._head = (self._head + 1) % self._capacity
        self._count -= 1
        return value
