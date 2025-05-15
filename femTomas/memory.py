from typing import List, Optional, Tuple
from .config import MEMORY_SIZE_WORDS, WORD_SIZE_BITS

class Memory:
    """Simulates the processor's main memory."""

    def __init__(self, initial_data: Optional[List[Tuple[int, int]]] = None):
        """
        Initializes the memory.

        Args:
            initial_data: A list of (address, value) tuples to pre-populate memory.
                          Addresses are word addresses.
        """
        # Max value for a 16-bit word
        self._max_word_value = (1 << WORD_SIZE_BITS) - 1
        self._min_word_value = -(1 << (WORD_SIZE_BITS - 1)) # For signed interpretation if needed, though typically memory stores unsigned bit patterns.
                                                            # For simplicity, we'll store as positive integers 0-65535 after masking.

        # Initialize memory with zeros
        self.data: List[int] = [0] * MEMORY_SIZE_WORDS

        if initial_data:
            for address, value in initial_data:
                self.write_word(address, value)

    def _validate_address(self, address: int) -> None:
        """Checks if the address is within the valid memory range."""
        if not (0 <= address < MEMORY_SIZE_WORDS):
            raise ValueError(
                f"Memory access error: Address {address} is out of bounds "
                f"(0-{MEMORY_SIZE_WORDS - 1})."
            )

    def _normalize_value(self, value: int) -> int:
        """Ensures the value fits within a 16-bit unsigned word (0-65535)."""
        return int(value) & self._max_word_value # Mask to 16 bits

    def read_word(self, address: int) -> int:
        """
        Reads a 16-bit word from the specified memory address.

        Args:
            address: The word address to read from.

        Returns:
            The 16-bit value stored at that address.

        Raises:
            ValueError: If the address is out of bounds.
        """
        self._validate_address(address)
        return self.data[address]

    def write_word(self, address: int, value: int) -> None:
        """
        Writes a 16-bit word to the specified memory address.
        The value will be masked to fit within 16 bits (0-65535).

        Args:
            address: The word address to write to.
            value: The 16-bit value to store.

        Raises:
            ValueError: If the address is out of bounds.
        """
        self._validate_address(address)
        normalized_value = self._normalize_value(value)
        self.data[address] = normalized_value

    def __str__(self) -> str:
        # Provides a string representation of a segment of memory, e.g., first few locations
        # or non-zero locations. For now, a simple summary.
        non_zero_count = sum(1 for x in self.data if x != 0)
        return f"Memory({MEMORY_SIZE_WORDS} words, {non_zero_count} non-zero entries)"

    def dump(self, start_address: int = 0, num_words: int = 16) -> List[Tuple[int, int]]:
        """Returns a list of (address, value) tuples for a specified memory range."""
        self._validate_address(start_address)
        end_address = min(start_address + num_words, MEMORY_SIZE_WORDS)
        if start_address >= end_address:
            return []
        return [(addr, self.data[addr]) for addr in range(start_address, end_address)]



# Tester

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from femTomas.config import MEMORY_SIZE_WORDS, WORD_SIZE_BITS # Now this should work

    # Example Usage and Basic Tests for Memory
    print(f"Memory Size: {MEMORY_SIZE_WORDS} words")
    mem = Memory(initial_data=[(0, 100), (1, 200), (MEMORY_SIZE_WORDS - 1, 65535)])

    print(f"Read from 0: {mem.read_word(0)}")
    print(f"Read from 1: {mem.read_word(1)}")

    mem.write_word(2, 300)
    print(f"Read from 2 (after write): {mem.read_word(2)}")

    mem.write_word(3, 70000) # Value too large, should be masked
    print(f"Read from 3 (after writing 70000): {mem.read_word(3)} (expected {70000 & 0xFFFF})")

    mem.write_word(4, -1) # Negative value, should be masked (becomes 65535)
    print(f"Read from 4 (after writing -1): {mem.read_word(4)} (expected {-1 & 0xFFFF})")

    print(f"Read from last address: {mem.read_word(MEMORY_SIZE_WORDS - 1)}")

    print("\nMemory dump (first 5 words):")
    for addr, val in mem.dump(0, 5):
        print(f"Addr {addr:04x}: {val:04x} ({val})")

    print("\nTesting out-of-bounds access:")
    try:
        mem.read_word(MEMORY_SIZE_WORDS)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    try:
        mem.write_word(-1, 10)
    except ValueError as e:
        print(f"Caught expected error: {e}")

    print(f"\nMemory object: {mem}")
