from typing import Optional, Any # Using Any for Instruction for now to avoid circular dependency
from .instruction import OpType
from .config import FU_CONFIG

class ReservationStation:
    """Represents a single entry in a reservation station pool for a functional unit type."""

    def __init__(self, name: str, fu_type: str, latency: int):
        """
        Initializes a reservation station entry.

        Args:
            name: Unique name/tag for this RS (e.g., "Load1", "AddSub3").
            fu_type: The type of Functional Unit this RS is associated with
                     (e.g., "LOAD", "ADD_SUB", "MUL"). Used to get latency.
            latency: The latency for this RS, from the processor's config.
        """
        self.name: str = name
        self.fu_type: str = fu_type
        self.latency: int = latency

        # State fields, reset by clear()
        self.busy: bool = False
        self.instruction: Optional[Any] = None # Will hold an 'Instruction' object
        self.op_type: Optional[OpType] = None

        self.Vj: Optional[int] = None  # Value of source operand 1
        self.Vk: Optional[int] = None  # Value of source operand 2
        self.Qj: Optional[str] = None  # RS tag producing Vj (if not ready)
        self.Qk: Optional[str] = None  # RS tag producing Vk (if not ready)

        self.A: Optional[int] = None   # Immediate value or effective address for LOAD/STORE

        self.remaining_execution_cycles: Optional[int] = None
        self.result_ready_for_cdb: bool = False # True when execution is complete

    def issue(
        self, 
        instruction_obj: Any, # Type hint to 'Instruction' later
        op: OpType, 
        vj_val: Optional[int], qj_tag: Optional[str],
        vk_val: Optional[int], qk_tag: Optional[str],
        imm_or_addr: Optional[int]
    ) -> None:
        """Populates the RS when an instruction is issued to it."""
        if self.busy:
            raise RuntimeError(f"Cannot issue to already busy RS: {self.name}")

        self.instruction = instruction_obj
        self.op_type = op
        self.busy = True
        self.result_ready_for_cdb = False

        self.Vj = vj_val
        self.Qj = qj_tag
        self.Vk = vk_val
        self.Qk = qk_tag
        self.A = imm_or_addr
        
        self.remaining_execution_cycles = self.latency

    def clear(self) -> None:
        """Resets the reservation station to be free and clear all fields."""
        self.busy = False
        self.instruction = None
        self.op_type = None
        self.Vj = None
        self.Vk = None
        self.Qj = None
        self.Qk = None
        self.A = None
        self.remaining_execution_cycles = None
        self.result_ready_for_cdb = False

    def is_ready_to_dispatch(self) -> bool:
        """
        Checks if all operands are available (Qj and Qk are None)
        and the RS is busy and not yet completed.
        This means it can be dispatched to its FU if the FU is free.
        """
        return self.busy and self.Qj is None and self.Qk is None and not self.result_ready_for_cdb

    def snoop_cdb(self, broadcasting_rs_tag: str, result_value: int) -> bool:
        """
        Monitors the Common Data Bus (CDB) for results.
        If this RS is waiting for the broadcasting_rs_tag, it captures the value.

        Args:
            broadcasting_rs_tag: The tag of the RS that has finished execution.
            result_value: The value produced.
        
        Returns:
            True if this RS captured a value, False otherwise.
        """
        updated = False
        if self.busy:
            if self.Qj == broadcasting_rs_tag:
                self.Vj = result_value
                self.Qj = None
                updated = True
            if self.Qk == broadcasting_rs_tag:
                self.Vk = result_value
                self.Qk = None
                updated = True
        return updated

    def execute_cycle(self) -> Optional[int]:
        """
        Simulates one cycle of execution in the FU.
        Decrements remaining_execution_cycles.
        If execution finishes, marks result_ready_for_cdb and returns the computed result.

        Returns:
            Computed result if execution finishes this cycle, else None.
            (For LOAD/STORE, result might be effective address or data from memory,
             this needs to be coordinated with the main Processor logic and Memory object).
             For now, this method is simplified and a placeholder for actual computation.
        """
        if not self.busy or self.remaining_execution_cycles is None or self.result_ready_for_cdb:
            return None

        if self.remaining_execution_cycles > 0:
            self.remaining_execution_cycles -= 1

        if self.remaining_execution_cycles == 0:
            self.result_ready_for_cdb = True
            # TODO: Actual computation of the result based on self.op_type, self.Vj, self.Vk, self.A
            # This is a placeholder. The actual calculation (e.g. Vj + Vk) 
            # and memory interaction for LOAD/STORE will happen in the Processor class
            # when this RS is selected for execution and completes.
            # For now, let's assume a dummy result if needed for testing.
            if self.instruction: # Accessing instruction.result_value as placeholder
                return self.instruction.result_value # This assumes result_value was pre-set or calculated elsewhere
            return 0 # Dummy result
        return None

    def get_result(self) -> Optional[int]:
        """Returns the computed result if ready. Placeholder for actual result retrieval."""
        if self.result_ready_for_cdb and self.instruction:
            # This is a placeholder. The actual result calculation logic is complex
            # and depends on the instruction type (ALU, Load, Store).
            # It will likely be managed by the Processor class which has access to memory etc.
            # For now, assume instruction.result_value is populated somehow during execute.
            return self.instruction.result_value 
        return None # Or a dummy value like 0 if result is ready but not yet computed

    def __str__(self) -> str:
        if not self.busy:
            return f"RS({self.name}, FU: {self.fu_type}): Free"
        
        qj_str = f"Val:{self.Vj}" if self.Qj is None else f"Tag:{self.Qj}"
        qk_str = f"Val:{self.Vk}" if self.Qk is None else f"Tag:{self.Qk}"
        
        timing_info = ""
        if self.instruction and self.instruction.issue_cycle is not None:
            timing_info = f", Issued@{self.instruction.issue_cycle}"
        if self.remaining_execution_cycles is not None:
            timing_info += f", RemExec:{self.remaining_execution_cycles}"
        if self.result_ready_for_cdb:
            timing_info += ", ResultReady"
            
        return (
            f"RS({self.name}, FU: {self.fu_type}, Busy: {self.busy}, Op: {self.op_type.name if self.op_type else 'N/A'}, "
            f"Instr: '{str(self.instruction)}', "
            f"Vj: {qj_str}, Vk: {qk_str}, A: {self.A}{timing_info})"
        )

if __name__ == '__main__':
    # Adjust sys.path for direct execution
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from femTomas.instruction import Instruction, OpType # For test instruction objects

    print("Testing Reservation Station")

    # Mock instruction objects for testing
    # In a real scenario, these would be proper Instruction instances
    mock_instr_add = Instruction("ADD R1, R2, R3", 0)
    mock_instr_add.op_type = OpType.ADD # Manually set for test
    mock_instr_add.rd = 1
    mock_instr_add.rs1 = 2
    mock_instr_add.rs2 = 3
    mock_instr_add.result_value = 15 # Dummy result for testing get_result

    mock_instr_load = Instruction("LOAD R4, 10(R5)", 1)
    mock_instr_load.op_type = OpType.LOAD
    mock_instr_load.rd = 4
    mock_instr_load.rs1 = 5 # Base register
    mock_instr_load.imm = 10 # Offset
    mock_instr_load.result_value = 100 # Dummy result

    # Create RS instances
    rs_add1 = ReservationStation(name="Add1", fu_type="ADD_SUB")
    rs_load1 = ReservationStation(name="Load1", fu_type="LOAD")
    print(f"Initial state: {rs_add1}")
    print(f"Initial state: {rs_load1}")

    # Test issue
    print("\nIssuing ADD to Add1 (R2=10, R3 from RS_LOAD1 which has tag 'Load1')")
    rs_add1.issue(mock_instr_add, OpType.ADD, vj_val=10, qj_tag=None, vk_val=None, qk_tag="Load1", imm_or_addr=None)
    mock_instr_add.issue_cycle = 1 # Simulate timing info being set
    print(rs_add1)
    assert rs_add1.busy
    assert rs_add1.Vj == 10 and rs_add1.Qj is None
    assert rs_add1.Vk is None and rs_add1.Qk == "Load1"
    assert not rs_add1.is_ready_to_dispatch()

    print("\nIssuing LOAD to Load1 (R5=20, offset=10)")
    rs_load1.issue(mock_instr_load, OpType.LOAD, vj_val=20, qj_tag=None, vk_val=None, qk_tag=None, imm_or_addr=10)
    mock_instr_load.issue_cycle = 1
    print(rs_load1)
    assert rs_load1.is_ready_to_dispatch() # Ready as R5 value is known, offset is A

    # Test execution cycle for Load1 (latency for LOAD is 6 from FU_CONFIG)
    print(f"\nExecuting Load1 (latency {rs_load1.latency} cycles):")
    for i in range(rs_load1.latency):
        print(f" Cycle {i+1}: Remaining_exec={rs_load1.remaining_execution_cycles}")
        result = rs_load1.execute_cycle()
        assert rs_load1.busy
        if result is not None:
            print(f"  Load1 finished execution. Result: {result} (dummy from mock_instr_load.result_value)")
            assert rs_load1.result_ready_for_cdb
            assert rs_load1.get_result() == 100
            break 
    print(rs_load1)

    # Test CDB snoop: Load1 broadcasts, Add1 is waiting for "Load1"
    print(f"\nLoad1 broadcasts result (100). Add1 snoops.")
    captured = rs_add1.snoop_cdb(broadcasting_rs_tag="Load1", result_value=100)
    print(f"Add1 after snoop: {rs_add1}")
    assert captured
    assert rs_add1.Vk == 100 and rs_add1.Qk is None
    assert rs_add1.is_ready_to_dispatch() # Now Add1 should be ready

    # Test execution cycle for Add1 (latency for ADD_SUB is 2)
    print(f"\nExecuting Add1 (latency {rs_add1.latency} cycles):")
    for i in range(rs_add1.latency):
        print(f" Cycle {i+1}: Remaining_exec={rs_add1.remaining_execution_cycles}")
        result = rs_add1.execute_cycle()
        if result is not None:
            print(f"  Add1 finished execution. Result: {result} (dummy from mock_instr_add.result_value)")
            assert rs_add1.result_ready_for_cdb
            assert rs_add1.get_result() == 15 # (10 + 5 for example, dummy here)
            break
    print(rs_add1)

    # Test clear
    print("\nClearing Add1:")
    rs_add1.clear()
    print(rs_add1)
    assert not rs_add1.busy

    print("\nAll Reservation Station tests seem to pass based on assertions.")
