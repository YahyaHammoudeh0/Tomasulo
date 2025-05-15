from typing import List, Optional, Dict
from .config import NUM_REGISTERS, WORD_SIZE_BITS

class RegisterFile:
    """Simulates the processor's register file and Register Result Status (RAT)."""

    def __init__(self):
        """Initializes registers to 0 and RAT entries to indicate values are ready in regs."""
        self._word_mask = (1 << WORD_SIZE_BITS) - 1

        # Physical registers R0-R7
        self.registers: List[int] = [0] * NUM_REGISTERS

        # Register Result Status (RAT) or Register Alias Table
        # Stores the tag (e.g., string ID like "RS_ADD1") of the Reservation Station
        # that will produce the result for the register.
        # None means the value in self.registers[reg_idx] is the latest.
        # R0 is constant, so no RAT entry needed, but for simplicity of indexing we include it.
        self.rat: List[Optional[str]] = [None] * NUM_REGISTERS

    def _normalize_value(self, value: int) -> int:
        """Ensures the value fits within a 16-bit unsigned word (0-65535)."""
        return int(value) & self._word_mask

    def read_physical_reg(self, reg_idx: int) -> int:
        """
        Reads the value directly from the physical register array.
        This is typically used when the RAT indicates the value is valid here.
        R0 always returns 0.
        """
        if not (0 <= reg_idx < NUM_REGISTERS):
            raise ValueError(f"Invalid register index: {reg_idx}")
        if reg_idx == 0:
            return 0
        return self.registers[reg_idx]

    def write_physical_reg(self, reg_idx: int, value: int) -> None:
        """
        Writes a value directly to a physical register. R0 is ignored.
        This is typically called by the CDB logic when a result is broadcast AND
        the RAT entry for this register matched the broadcasting RS tag.
        The RAT entry should be cleared by the caller or CDB logic *after* this write.
        """
        if not (0 <= reg_idx < NUM_REGISTERS):
            raise ValueError(f"Invalid register index: {reg_idx}")
        if reg_idx == 0: # R0 is read-only, always 0
            return
        self.registers[reg_idx] = self._normalize_value(value)

    def get_rat_tag(self, reg_idx: int) -> Optional[str]:
        """
        Gets the RAT tag for a given register.
        Returns None if the register value is current in the physical register file.
        Returns the RS tag otherwise.
        R0 always effectively has a None tag (value is always ready).
        """
        if not (0 <= reg_idx < NUM_REGISTERS):
            raise ValueError(f"Invalid register index: {reg_idx}")
        if reg_idx == 0:
            return None
        return self.rat[reg_idx]

    def set_rat_tag(self, reg_idx: int, rs_tag: str) -> None:
        """
        Sets the RAT tag for a register, indicating it's waiting for rs_tag to produce its value.
        Does nothing for R0.
        """
        if not (0 <= reg_idx < NUM_REGISTERS):
            raise ValueError(f"Invalid register index: {reg_idx}")
        if reg_idx == 0:
            return
        if not rs_tag: # Should always be a valid tag
            raise ValueError("Cannot set an empty or None RS tag in RAT.")
        self.rat[reg_idx] = rs_tag

    def clear_rat_tag(self, reg_idx: int) -> None:
        """
        Clears the RAT tag for a register, indicating its value is now in the physical register.
        Typically called after a value is written to the physical register via CDB.
        """
        if not (0 <= reg_idx < NUM_REGISTERS):
            raise ValueError(f"Invalid register index: {reg_idx}")
        if reg_idx == 0:
            return
        self.rat[reg_idx] = None

    def on_broadcast(self, broadcasting_rs_tag: str, result_value: int) -> List[int]:
        """
        Called when a result is broadcast on the CDB.
        Updates any physical register whose RAT entry matches broadcasting_rs_tag.
        Clears the RAT entry for those registers.

        Args:
            broadcasting_rs_tag: The tag of the RS that has finished execution.
            result_value: The value produced.

        Returns:
            A list of register indices that were updated.
        """
        updated_regs = []
        # R0 is never updated by RAT logic
        for i in range(1, NUM_REGISTERS):
            if self.rat[i] == broadcasting_rs_tag:
                self.write_physical_reg(i, result_value)
                self.clear_rat_tag(i)
                updated_regs.append(i)
        return updated_regs

    def __str__(self) -> str:
        reg_strs = []
        for i in range(NUM_REGISTERS):
            val = self.read_physical_reg(i)
            tag = self.get_rat_tag(i)
            reg_strs.append(f"R{i}: {val:04x} ({val}){' (Pending: ' + tag + ')' if tag else ''}")
        return "RegisterFile:\n  " + "\n  ".join(reg_strs)

if __name__ == '__main__':
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from femTomas.config import NUM_REGISTERS, WORD_SIZE_BITS # Now this should work

    # Example Usage and Basic Tests for RegisterFile
    rf = RegisterFile()
    print(rf)

    print("\nSimulating instruction issue: ADD R1, R2, R3 (R1 is dest)")
    # R2, R3 are available in RF. R1 will be written by "RS_ADD1"
    rf.set_rat_tag(1, "RS_ADD1")
    print(rf)
    print(f"R1 RAT tag: {rf.get_rat_tag(1)}") # Expected: RS_ADD1
    print(f"R2 value: {rf.read_physical_reg(2)}, R2 RAT tag: {rf.get_rat_tag(2)}") # Expected: 0, None

    print("\nSimulating instruction issue: LOAD R2, 0(R0) (R2 is dest)")
    # R0 is source (always 0). R2 will be written by "RS_LOAD1"
    rf.set_rat_tag(2, "RS_LOAD1")
    print(rf)
    print(f"R2 RAT tag: {rf.get_rat_tag(2)}") # Expected: RS_LOAD1

    print("\nSimulating broadcast from RS_ADD1 (result 12345 for R1):")
    updated_by_add = rf.on_broadcast("RS_ADD1", 12345)
    print(f"Registers updated by RS_ADD1 broadcast: {updated_by_add}")
    print(rf)
    print(f"R1 value: {rf.read_physical_reg(1)}, R1 RAT tag: {rf.get_rat_tag(1)}") # Expected: 12345, None

    print("\nSimulating broadcast from RS_LOAD1 (result 54321 for R2):")
    updated_by_load = rf.on_broadcast("RS_LOAD1", 54321)
    print(f"Registers updated by RS_LOAD1 broadcast: {updated_by_load}")
    print(rf)
    print(f"R2 value: {rf.read_physical_reg(2)}, R2 RAT tag: {rf.get_rat_tag(2)}") # Expected: 54321, None

    print("\nTesting R0 (should always be 0 and no RAT tag):")
    rf.set_rat_tag(0, "RS_FAKE")
    print(f"R0 value: {rf.read_physical_reg(0)}, R0 RAT tag: {rf.get_rat_tag(0)}") # Expected: 0, None
    rf.write_physical_reg(0, 999)
    print(f"R0 value after write attempt: {rf.read_physical_reg(0)}") # Expected: 0

    print("\nTesting invalid register access:")
    try:
        rf.read_physical_reg(NUM_REGISTERS)
    except ValueError as e:
        print(f"Caught expected error for read: {e}")
    try:
        rf.set_rat_tag(NUM_REGISTERS, "RS_ERR")
    except ValueError as e:
        print(f"Caught expected error for set_rat_tag: {e}")
