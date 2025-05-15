from typing import List, Dict, Optional, Tuple, Any

from .config import FU_CONFIG, NUM_REGISTERS, MEMORY_SIZE_WORDS
from .instruction import Instruction, OpType
from .memory import Memory
from .register_file import RegisterFile
from .reservation_station import ReservationStation

class Processor:
    """Orchestrates the Tomasulo algorithm simulation."""

    def __init__(self, fu_config=None, pipeline_width=None):
        from . import config as global_config
        self.fu_config = fu_config if fu_config is not None else global_config.FU_CONFIG
        self.pipeline_width = pipeline_width if pipeline_width is not None else getattr(global_config, 'PIPELINE_WIDTH', 1)

        self.memory = Memory()
        self.register_file = RegisterFile()
        
        self.reservation_stations: Dict[str, List[ReservationStation]] = {}
        self._initialize_reservation_stations()

        self.instruction_queue: List[Instruction] = []
        self.loaded_instructions_map: Dict[int, Instruction] = {} # Map address to Instruction
        self.program_counter: int = 0 # Word address
        
        self.current_cycle: int = 0

        # To store timing information for each instruction (keyed by instruction unique ID or initial PC)
        # Example: {instr_uid: {'I': cycle, 'ES': cycle, 'EE': cycle, 'WB': cycle, 'RAW': 'instr_str'}}
        self.timing_log: Dict[int, Dict[str, Any]] = {}
        self._next_instr_uid = 0 # For uniquely identifying instructions for logging

        # CDB state: (tag_of_broadcasting_rs, result_value)
        self.cdb_broadcast_this_cycle: Optional[Tuple[str, int]] = None

    def _initialize_reservation_stations(self):
        """Creates RS instances based on self.fu_config."""
        self.reservation_stations = {}
        for fu_type, config in self.fu_config.items():
            self.reservation_stations[fu_type] = []
            for i in range(config['rs_count']):
                rs_name = f"{fu_type}{i+1}"
                self.reservation_stations[fu_type].append(ReservationStation(name=rs_name, fu_type=fu_type))

    def load_program(
        self, 
        program_string: str, 
        initial_pc: int = 0, 
        initial_memory_data: Optional[List[Tuple[int, int]]] = None,
        initial_register_data: Optional[List[Tuple[int, int]]] = None
    ):
        """
        Loads a program from a string, sets initial PC, memory, and registers.
        Each line in program_string is an assembly instruction.
        Memory addresses for instructions are assumed to be contiguous starting from initial_pc.
        """
        self.program_counter = initial_pc
        self.current_cycle = 0
        self.instruction_queue = []
        self.loaded_instructions_map = {}
        self.timing_log = {}
        self._next_instr_uid = 0
        self.cdb_broadcast_this_cycle = None

        # Initialize memory and registers
        self.memory = Memory(initial_data=initial_memory_data)
        self.register_file = RegisterFile()
        if initial_register_data:
            for reg_idx, val in initial_register_data:
                if reg_idx != 0: # R0 is not writable
                    self.register_file.write_physical_reg(reg_idx, val)
                    self.register_file.clear_rat_tag(reg_idx)

        # Clear all reservation stations
        for fu_type_list in self.reservation_stations.values():
            for rs in fu_type_list:
                rs.clear()

        # First pass: build label->addr map
        label_map = {}
        lines = program_string.strip().split('\n')
        current_instr_addr = initial_pc
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines or comments
                continue
            
            if ':' in line:
                label_name = line.split(':')[0]
                label_map[label_name] = current_instr_addr
                line = line.split(':', 1)[1].strip()
            
            current_instr_addr += 1
        
        # Second pass: parse instructions with label resolution
        current_instr_addr = initial_pc
        for line in lines:
            line = line.strip()
            if not line or line.startswith('#'): # Skip empty lines or comments
                continue
            
            if ':' in line:
                line = line.split(':', 1)[1].strip()
            
            instr_uid = self._next_instr_uid
            self._next_instr_uid += 1
            
            try:
                instr = Instruction(raw_instruction=line, address=current_instr_addr, uid=instr_uid)
                self.instruction_queue.append(instr)
                self.loaded_instructions_map[current_instr_addr] = instr
                self.timing_log[instr.uid] = {'RAW': str(instr), 'I': None, 'ES': None, 'EE': None, 'WB': None}
                current_instr_addr += 1 # Assuming each instruction takes 1 word, PC increments by 1
            except ValueError as e:
                print(f"Error parsing instruction '{line}' at address {current_instr_addr}: {e}")
                # Decide on error handling: stop loading, or skip instruction?
                # For now, let's skip and print error.

        print(f"Program loaded. {len(self.instruction_queue)} instructions. PC set to {self.program_counter}.")
        # print("Initial Timing Log:", self.timing_log)

    def _get_free_rs(self, fu_type: str) -> Optional[ReservationStation]:
        """Finds a free RS for the given functional unit type."""
        if fu_type in self.reservation_stations:
            for rs in self.reservation_stations[fu_type]:
                if not rs.busy:
                    return rs
        return None

    def _issue_stage(self):
        """Handles the instruction issue stage. Supports multiple-issue."""
        issued_this_cycle = 0
        while issued_this_cycle < self.pipeline_width:
            if self.program_counter not in self.loaded_instructions_map:
                break # No more instructions to issue or PC is out of sync

            instr_to_issue = self.loaded_instructions_map[self.program_counter]

            # Prevent re-issuing an already issued (or processing) instruction
            if self.timing_log[instr_to_issue.uid]['I'] is not None:
                break # Can't issue same instruction twice; stop issuing this cycle

            fu_type_needed = instr_to_issue.get_fu_type()
            if fu_type_needed is None:
                print(f"Cycle {self.current_cycle}: Cannot determine FU type for {instr_to_issue}. Skipping issue.")
                break

            rs = self._get_free_rs(fu_type_needed)
            if not rs:
                break # Structural hazard; stop issuing this cycle

            # Read source operands and RAT status
            vj, qj = None, None
            if instr_to_issue.rs1 is not None:
                rat_tag_j = self.register_file.get_rat_tag(instr_to_issue.rs1)
                if rat_tag_j:
                    qj = rat_tag_j
                else:
                    vj = self.register_file.read_physical_reg(instr_to_issue.rs1)

            vk, qk = None, None
            if instr_to_issue.op_type not in [OpType.LOAD, OpType.JAL, OpType.BEQ, OpType.JMP]:
                if instr_to_issue.rs2 is not None:
                    rat_tag_k = self.register_file.get_rat_tag(instr_to_issue.rs2)
                    if rat_tag_k:
                        qk = rat_tag_k
                    else:
                        vk = self.register_file.read_physical_reg(instr_to_issue.rs2)
            
            # Handle immediate for LOAD/STORE/ADDI, etc.
            A_val = instr_to_issue.imm 

            # Issue to RS
            rs.issue(instr_to_issue, instr_to_issue.op_type, vj, qj, vk, qk, A_val)
            
            # Update RAT for the destination register (if any)
            if instr_to_issue.rd is not None and instr_to_issue.op_type not in [OpType.STORE, OpType.BEQ, OpType.JMP]:
                self.register_file.set_rat_tag(instr_to_issue.rd, rs.name)

            # Log issue time and advance PC
            self.timing_log[instr_to_issue.uid]['I'] = self.current_cycle
            instr_to_issue.issue_cycle = self.current_cycle
            print(f"Cycle {self.current_cycle}: Issued {instr_to_issue} to {rs.name}")
            self.program_counter += 1 # Advance PC to next instruction's address

            issued_this_cycle += 1

    def _execute_stage(self):
        """Handles the instruction execution stage."""
        # This stage is complex: dispatching ready RSs to FUs, and FU execution cycles.
        # For simplicity, we'll assume FUs can start new operations if an RS is ready
        # and the FU itself is not busy with a multi-cycle operation from a *previous* cycle.
        # The ReservationStation's execute_cycle handles its internal timing.

        for fu_type, rs_list in self.reservation_stations.items():
            for rs in rs_list:
                if rs.busy and not rs.result_ready_for_cdb:
                    # If RS is ready to dispatch and its FU is conceptually 'free'
                    # (for now, we assume an RS with remaining_execution_cycles > 0 is 'using' the FU)
                    if rs.is_ready_to_dispatch() and self.timing_log[rs.instruction.uid]['ES'] is None:
                        self.timing_log[rs.instruction.uid]['ES'] = self.current_cycle # Execution Start
                        rs.instruction.execute_start_cycle = self.current_cycle
                        print(f"Cycle {self.current_cycle}: {rs.name} ({rs.instruction}) started execution.")

                    # If instruction has started execution, advance its execution cycle
                    if self.timing_log[rs.instruction.uid]['ES'] is not None:
                        # Perform actual computation if it's the last cycle of execution
                        if rs.remaining_execution_cycles == 1: # About to finish
                            computed_result = self._compute_result(rs)
                            rs.instruction.result_value = computed_result # Store for broadcast
                        
                        rs.execute_cycle() # Decrements remaining_execution_cycles

                        if rs.result_ready_for_cdb:
                            self.timing_log[rs.instruction.uid]['EE'] = self.current_cycle # Execution End
                            rs.instruction.execute_end_cycle = self.current_cycle
                            print(f"Cycle {self.current_cycle}: {rs.name} ({rs.instruction}) finished execution. Result: {rs.instruction.result_value}")
    
    def _compute_result(self, rs: ReservationStation) -> int:
        """Computes the actual result for an instruction in an RS. Handles memory for L/S."""
        instr = rs.instruction
        op_type = rs.op_type
        val_j = rs.Vj
        val_k = rs.Vk
        imm_A = rs.A # This is offset for LOAD/STORE, immediate for ADDI

        # Ensure values are not None if expected (should be guaranteed by is_ready_to_dispatch)
        # This is a simplified placeholder for actual computation logic.
        result = 0
        if op_type == OpType.LOAD:
            # Effective address = R[rs1] + offset. rs.Vj holds R[rs1], rs.A holds offset
            eff_addr = (val_j if val_j is not None else 0) + (imm_A if imm_A is not None else 0)
            result = self.memory.read_word(eff_addr)
            print(f"Cycle {self.current_cycle}: {rs.name} (LOAD) calculated eff_addr={eff_addr}, read val={result}")
        elif op_type == OpType.STORE:
            # Effective address = R[rs2] + offset. rs.Vk holds R[rs2], rs.A holds offset
            # Value to store comes from R[rd] (or rs1 for some ISAs). Here, rs.Vj should hold value from rd/rs1.
            eff_addr = (val_k if val_k is not None else 0) + (imm_A if imm_A is not None else 0)
            value_to_store = val_j # Assuming Vj holds the value from the register to be stored (instr.rd)
            self.memory.write_word(eff_addr, value_to_store)
            result = value_to_store # STORE might not produce a "result" for RAT, but can be value stored for logging
            print(f"Cycle {self.current_cycle}: {rs.name} (STORE) calculated eff_addr={eff_addr}, wrote val={value_to_store}")
        elif op_type == OpType.ADD:
            result = (val_j if val_j is not None else 0) + (val_k if val_k is not None else 0)
        elif op_type == OpType.ADDI:
            result = (val_j if val_j is not None else 0) + (imm_A if imm_A is not None else 0)
        elif op_type == OpType.SUB:
            result = (val_j if val_j is not None else 0) - (val_k if val_k is not None else 0)
        elif op_type == OpType.MUL:
            result = (val_j if val_j is not None else 0) * (val_k if val_k is not None else 0)
        elif op_type == OpType.DIV:
            if (val_k if val_k is not None else 0) == 0:
                print(f"Cycle {self.current_cycle}: {rs.name} (DIV) Division by zero! Resulting in 0.")
                result = 0 # Handle division by zero
            else:
                result = int((val_j if val_j is not None else 0) / (val_k if val_k is not None else 0))
        # ... other ops like NAND, JMP, BEQ, JAL to be handled.
        # Jumps/Branches modify PC, handled differently.
        else:
            print(f"Cycle {self.current_cycle}: {rs.name} unknown op_type {op_type} for computation.")
            result = 0 # Default result

        return self.register_file._normalize_value(result) # Normalize to 16-bit

    def _write_back_stage(self):
        """Handles the write-back stage via the CDB."""
        self.cdb_broadcast_this_cycle = None # Clear CDB from previous cycle
        
        # Find one RS ready to broadcast (simple model: one CDB, FCFS or priority)
        # For now, just iterate and pick the first one found. Could be more sophisticated.
        broadcasting_rs: Optional[ReservationStation] = None
        for fu_type, rs_list in self.reservation_stations.items():
            for rs in rs_list:
                if rs.busy and rs.result_ready_for_cdb and self.timing_log[rs.instruction.uid]['WB'] is None:
                    # Check if this RS has not already broadcast (e.g. if WB takes a cycle)
                    # Or if this is the cycle it *becomes* ready to WB.
                    if self.cdb_broadcast_this_cycle is None: # CDB is free
                        broadcasting_rs = rs
                        break
            if broadcasting_rs:
                break
        
        if broadcasting_rs:
            rs_tag = broadcasting_rs.name
            result_val = broadcasting_rs.instruction.result_value # Value computed in _execute_stage
            
            self.cdb_broadcast_this_cycle = (rs_tag, result_val)
            self.timing_log[broadcasting_rs.instruction.uid]['WB'] = self.current_cycle
            broadcasting_rs.instruction.write_back_cycle = self.current_cycle
            print(f"Cycle {self.current_cycle}: CDB Broadcasting: {rs_tag} with result {result_val} ({broadcasting_rs.instruction})")

            # Update Register File and RAT
            # The on_broadcast method updates physical regs if RAT matches, and clears RAT tag.
            # This handles the case where rd of broadcasting_rs is waiting for this result.
            self.register_file.on_broadcast(rs_tag, result_val)
            
            # Snoop CDB in other Reservation Stations
            for fu_list in self.reservation_stations.values():
                for other_rs in fu_list:
                    if other_rs.busy and other_rs != broadcasting_rs: # Don't snoop self
                        if other_rs.snoop_cdb(rs_tag, result_val):
                            print(f"Cycle {self.current_cycle}: {other_rs.name} snooped {rs_tag} with value {result_val}")
            
            # CRITICAL: Clear the broadcasting RS so it can be reused.
            broadcasting_rs.clear()

    def run_cycle(self):
        """Simulates a single clock cycle."""
        self.current_cycle += 1
        print(f"--- Cycle {self.current_cycle} Start ---")
        
        # Stages are processed in specific order to reflect data flow in a real pipeline
        # Write Back -> Execute -> Issue
        # This allows results written back in this cycle to be available for execution/issue in the *next* cycle.
        # Or, if snoop happens before issue/execute logic within same cycle, it can work too.
        # For Tomasulo, CDB broadcast should be visible to RSs for operand capture quickly.

        self._write_back_stage()  # Determine what's broadcast on CDB this cycle
        self._execute_stage()     # Advance execution, compute results for those finishing
        self._issue_stage()       # Issue new instructions if possible

        print(f"Register File at end of Cycle {self.current_cycle}:")
        print(self.register_file)
        print(f"--- Cycle {self.current_cycle} End ---")

    def is_simulation_complete(self) -> bool:
        """Checks if all instructions have been issued, executed, and written back."""
        if not self.instruction_queue: # No program loaded
            return True 
        
        # Check if PC is beyond the last loaded instruction's address AND all issued instructions have WB
        # A simpler check: all entries in timing_log have a WB cycle.
        for instr_uid, times in self.timing_log.items():
            if times['WB'] is None:
                return False # At least one instruction hasn't completed Write Back
        
        # Also ensure no RS is busy (especially for instructions like STORE that might not update RAT directly)
        for fu_list in self.reservation_stations.values():
            for rs in fu_list:
                if rs.busy:
                    return False
        
        return True

    def run_simulation(self, max_cycles=1000):
        """Runs the simulation until completion or max_cycles."""
        print("Starting simulation...")
        while not self.is_simulation_complete() and self.current_cycle < max_cycles:
            self.run_cycle()
        
        if self.current_cycle >= max_cycles:
            print(f"Simulation stopped at max cycles: {max_cycles}")
        else:
            print(f"Simulation completed in {self.current_cycle} cycles.")
        self.print_timing_results()

    def print_timing_results(self):
        """Prints the instruction timing log in a formatted way."""
        print("\n--- Instruction Timing Results ---")
        # Sort by issue cycle or original instruction order (UID)
        # Header: Instruction | Issue | ExecStart | ExecEnd | WriteBack
        print(f"{'Instruction':<30} | {'UID':>3} | {'Issue (I)':>10} | {'ExecStart (ES)':>15} | {'ExecEnd (EE)':>14} | {'WriteBack (WB)':>15}")
        print("-" * 100)
        
        sorted_instr_uids = sorted(self.timing_log.keys())

        for uid in sorted_instr_uids:
            log = self.timing_log[uid]
            instr_str = log.get('RAW', 'N/A')
            i_cycle = log.get('I', '-')
            es_cycle = log.get('ES', '-')
            ee_cycle = log.get('EE', '-')
            wb_cycle = log.get('WB', '-')
            print(f"{instr_str:<30} | {uid:>3} | {str(i_cycle):>10} | {str(es_cycle):>15} | {str(ee_cycle):>14} | {str(wb_cycle):>15}")
        print("--- End of Timing Results ---")

# Example Usage (to be refined and expanded)
if __name__ == '__main__':
    # Adjust sys.path for direct execution if necessary, though Processor should manage its imports
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    import femTomas.config as config

    def get_user_hardware_config():
        print("=== Tomasulo Simulator Hardware Configuration ===")
        default_fu_config = {
            "LOAD":     {"rs_count": 2, "latency": 6},
            "STORE":    {"rs_count": 2, "latency": 6},
            "BEQ":      {"rs_count": 2, "latency": 1},
            "CALL":     {"rs_count": 1, "latency": 1},
            "RET":      {"rs_count": 1, "latency": 1},
            "ADD_SUB":  {"rs_count": 4, "latency": 2},
            "NOR":      {"rs_count": 2, "latency": 1},
            "MUL":      {"rs_count": 2, "latency": 10},
        }
        print("Press Enter to accept the default value shown in [brackets].")
        fu_config = {}
        for fu, vals in default_fu_config.items():
            while True:
                try:
                    rs_count = input(f"Number of reservation stations for {fu} [{vals['rs_count']}]: ")
                    rs_count = int(rs_count) if rs_count.strip() else vals['rs_count']
                    latency = input(f"Cycles needed for {fu} [{vals['latency']}]: ")
                    latency = int(latency) if latency.strip() else vals['latency']
                    fu_config[fu] = {"rs_count": rs_count, "latency": latency}
                    break
                except ValueError:
                    print("Please enter a valid integer.")
        while True:
            try:
                pipeline_width = input(f"Pipeline width (instructions issued per cycle) [1]: ")
                pipeline_width = int(pipeline_width) if pipeline_width.strip() else 1
                break
            except ValueError:
                print("Please enter a valid integer.")
        print("\nHardware configuration complete!\n")
        return fu_config, pipeline_width

    # Get user hardware config interactively
    user_fu_config, user_pipeline_width = get_user_hardware_config()

    processor = Processor(fu_config=user_fu_config, pipeline_width=user_pipeline_width)
    
    while True:
        print("\nChoose a program input method:")
        print("1. Manual input")
        print("2. Load from file")
        choice = input("Enter your choice (1/2): ")
        if choice == "1":
            print("Enter your program (assembly instructions, one per line). Type 'END' on a new line to finish:")
            lines = []
            while True:
                line = input()
                if line.strip().upper() == 'END':
                    break
                lines.append(line)
            program = '\n'.join(lines)
        elif choice == "2":
            filename = input("Enter the filename of your program: ")
            with open(filename, 'r') as f:
                program = f.read()
        else:
            print("Invalid choice. Please try again.")
            continue
        
        processor.load_program(program, initial_pc=0)
        processor.run_simulation(max_cycles=50)

        print("\nFinal Register File State:")
        print(processor.register_file)
        print("\nFinal Memory State (first few words relevant to program):")
        print("Relevant memory dump (e.g., address 30):")
        print(processor.memory.dump(25, 10))
