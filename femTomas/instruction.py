import re
from enum import Enum, auto
from typing import Optional

class OpType(Enum):
    LOAD = auto()    # Load: LD Rdest, offset(Rbase)
    STORE = auto()   # Store: SD Rsrc, offset(Rbase)
    ADD = auto()     # Add: ADD Rdest, Rsrc1, Rsrc2
    ADDI = auto()    # Add Immediate: ADDI Rdest, Rsrc1, Imm
    SUB = auto()     # Subtract: SUB Rdest, Rsrc1, Rsrc2
    MUL = auto()     # Multiply: MUL Rdest, Rsrc1, Rsrc2
    DIV = auto()     # Divide: DIV Rdest, Rsrc1, Rsrc2
    NAND = auto()    # NAND: NAND Rdest, Rsrc1, Rsrc2
    JMP = auto()     # Unconditional Jump: JMP Imm (offset from PC)
    BEQ = auto()     # Branch if Equal: BEQ Rs1, Rs2, Imm (offset from PC)
    JAL = auto()     # Jump and Link: JAL Rdest, Imm (offset from PC)
    RET = auto()     # Return from subroutine (special case of JALR often, or JMP) - simplified as JMP to R[rd] if R[rd] is link
    NOP = auto()     # No operation
    HALT = auto()    # Halt execution (can be a special NOP or specific instruction)

class Instruction:
    """
    Represents a parsed assembly instruction.
    """
    def __init__(self, raw_instruction: str, address: int, uid: int):
        self.raw_instruction: str = raw_instruction.strip()
        self.address: int = address  # Memory address of this instruction
        self.uid: int = uid          # Unique ID for this instruction instance

        self.op_type: Optional[OpType] = None
        self.rd: Optional[int] = None  # Destination register index
        self.rs1: Optional[int] = None # Source register 1 index
        self.rs2: Optional[int] = None # Source register 2 index
        self.imm: Optional[int] = None # Immediate value or offset

        # Timing information to be filled by the processor
        self.issue_cycle: Optional[int] = None
        self.execute_start_cycle: Optional[int] = None
        self.execute_end_cycle: Optional[int] = None
        self.write_back_cycle: Optional[int] = None
        self.result_value: Optional[int] = None # To store computed result before WB

        try:
            self._parse()
        except ValueError as e:
            # Re-raise with more context or handle as an invalid instruction
            raise ValueError(f"Failed to parse instruction '{self.raw_instruction}': {e}")

    def _parse_register(self, reg_str: str) -> int:
        """Converts a register string like 'R1' to an integer index 1."""
        match = re.fullmatch(r"R([0-7])", reg_str, re.IGNORECASE)
        if not match:
            raise ValueError(f"Invalid register format: '{reg_str}'")
        return int(match.group(1))

    def _parse_immediate(self, imm_str: str) -> int:
        """Converts an immediate string to an integer."""
        try:
            return int(imm_str)
        except ValueError:
            raise ValueError(f"Invalid immediate value: '{imm_str}'")

    def _parse(self):
        # Clean the instruction string by removing comments and extra whitespace
        instruction_to_parse = self.raw_instruction.split('#')[0].strip()

        parts = re.split(r"[\s,]+", instruction_to_parse, 1)
        op_str = parts[0].upper()
        # Ensure operands_str is also stripped, in case it's empty or just whitespace after comment removal
        operands_str = (parts[1] if len(parts) > 1 else "").strip()

        # R-type: OP Rdest, Rsrc1, Rsrc2 (e.g., ADD, SUB, MUL, DIV, NAND)
        r_type_ops = {
            "ADD": OpType.ADD, "SUB": OpType.SUB, "MUL": OpType.MUL,
            "DIV": OpType.DIV, "NAND": OpType.NAND
        }
        # I-type (arithmetic/load): OP Rdest, Rsrc1, Imm (e.g., ADDI, LOAD)
        i_type_arith_load_ops = {"ADDI": OpType.ADDI, "LOAD": OpType.LOAD}
        # S-type (store): OP Rsrc1, Imm(Rsrc2) (e.g., STORE Rvalue, offset(Rbase))
        # B-type (branch): OP Rsrc1, Rsrc2, Imm (e.g., BEQ)
        # J-type (jump): OP Imm (e.g., JMP) or OP Rdest, Imm (e.g., JAL)

        if op_str in r_type_ops:
            self.op_type = r_type_ops[op_str]
            match = re.fullmatch(r"(R[0-7]),\s*(R[0-7]),\s*(R[0-7])", operands_str, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid R-type format for {op_str}: '{operands_str}'")
            self.rd = self._parse_register(match.group(1))
            self.rs1 = self._parse_register(match.group(2))
            self.rs2 = self._parse_register(match.group(3))
        elif op_str in i_type_arith_load_ops:
            self.op_type = i_type_arith_load_ops[op_str]
            if self.op_type == OpType.LOAD: # LD Rdest, offset(Rbase)
                match = re.fullmatch(r"(R[0-7]),\s*(-?\d+)\((R[0-7])\)", operands_str, re.IGNORECASE)
                if not match:
                    raise ValueError(f"Invalid LOAD format: '{operands_str}'")
                self.rd = self._parse_register(match.group(1))
                self.imm = self._parse_immediate(match.group(2))
                self.rs1 = self._parse_register(match.group(3)) # Base register
            else: # ADDI Rdest, Rsrc1, Imm
                match = re.fullmatch(r"(R[0-7]),\s*(R[0-7]),\s*(-?\d+)", operands_str, re.IGNORECASE)
                if not match:
                    raise ValueError(f"Invalid ADDI format: '{operands_str}'")
                self.rd = self._parse_register(match.group(1))
                self.rs1 = self._parse_register(match.group(2))
                self.imm = self._parse_immediate(match.group(3))
        elif op_str == "STORE": # SD Rsrc, offset(Rbase) -> we'll map Rsrc to rd for consistency internally if needed, rs1 to Rbase
            self.op_type = OpType.STORE
            match = re.fullmatch(r"(R[0-7]),\s*(-?\d+)\((R[0-7])\)", operands_str, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid STORE format: '{operands_str}'")
            self.rs1 = self._parse_register(match.group(1)) # Value to store (from rs1)
            self.imm = self._parse_immediate(match.group(2)) # Offset
            self.rs2 = self._parse_register(match.group(3)) # Base register
            # For STORE, rd is not used for destination, but rs1 holds the value source reg.
        elif op_str == "JMP":
            self.op_type = OpType.JMP
            self.imm = self._parse_immediate(operands_str) # PC-relative offset
        elif op_str == "BEQ": # BEQ Rsrc1, Rsrc2, Imm
            self.op_type = OpType.BEQ
            match = re.fullmatch(r"(R[0-7]),\s*(R[0-7]),\s*(-?\d+)", operands_str, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid BEQ format: '{operands_str}'")
            self.rs1 = self._parse_register(match.group(1))
            self.rs2 = self._parse_register(match.group(2))
            self.imm = self._parse_immediate(match.group(3)) # PC-relative offset
        elif op_str == "JAL": # JAL Rdest, Imm
            self.op_type = OpType.JAL
            match = re.fullmatch(r"(R[0-7]),\s*(-?\d+)", operands_str, re.IGNORECASE)
            if not match:
                raise ValueError(f"Invalid JAL format: '{operands_str}'")
            self.rd = self._parse_register(match.group(1)) # Link register
            self.imm = self._parse_immediate(match.group(2)) # PC-relative offset
        elif op_str == "RET": # Simplified: JMP R[link_reg], usually R1 or a convention
                              # For Tomasulo, might be more complex or just a JMP.
                              # Let's treat it as JMP to address in R1 (convention)
            self.op_type = OpType.JMP # Or OpType.RET if we want specific FU
            # This interpretation of RET as JMP R1 is a simplification.
            # A true RET might use a dedicated stack or link register with specific hardware support.
            # For now, assume RET implies jumping to an address previously stored in a specific register (e.g., R1).
            # The simulator would need to know which register holds the return address.
            # We'll make it JMP to immediate 0 for now, or require specific RET syntax.
            # For now, this RET is more like a NOP or needs a specific FU and operand.
            # Let's make it JMP 0(R1) effectively, so rs1 = R1, imm = 0, no rd
            self.rs1 = 1 # Assume R1 holds return address by convention
            self.imm = 0 # Offset 0
            # This is a placeholder; RET behavior is ISA-dependent.
        elif op_str == "NOP":
            self.op_type = OpType.NOP
        elif op_str == "HALT":
            self.op_type = OpType.HALT
        else:
            raise ValueError(f"Unknown operation: '{op_str}'")

    def get_fu_type(self) -> Optional[str]:
        """Returns the type of Functional Unit required by this instruction."""
        if self.op_type in [OpType.LOAD, OpType.STORE]:
            return "LOAD" # Load/Store unit
        elif self.op_type in [OpType.ADD, OpType.SUB, OpType.ADDI]:
            return "ADD_SUB"
        elif self.op_type == OpType.MUL: 
            return "MUL"                
        elif self.op_type == OpType.DIV:
            return "DIV" 
        elif self.op_type == OpType.NAND:
            return "NAND" # Dedicated NAND unit or general ALU
        elif self.op_type in [OpType.JMP, OpType.BEQ, OpType.JAL, OpType.RET]:
            return "BRANCH" # Branch/Jump unit
        elif self.op_type in [OpType.NOP, OpType.HALT]:
            return None # NOP/HALT might not need a specific FU in the same way
        return None # Default for unhandled or NOP-like instructions

    def __str__(self) -> str:
        return self.raw_instruction

    def __repr__(self) -> str:
        details = [f"'{self.raw_instruction}' (Addr:{self.address}, UID:{self.uid}) Op:{self.op_type.name if self.op_type else 'N/A'}"]
        if self.rd is not None: details.append(f"Rd:R{self.rd}")
        if self.rs1 is not None: details.append(f"Rs1:R{self.rs1}")
        if self.rs2 is not None: details.append(f"Rs2:R{self.rs2}")
        if self.imm is not None: details.append(f"Imm:{self.imm}")
        return "<Instruction " + ", ".join(details) + ">"

if __name__ == '__main__':
    # Adjust sys.path for direct execution if necessary (for config import by other modules)
    import sys
    import os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    instructions_to_test = [
        ("LOAD R1, 100(R2)", 0, 0), ("STORE R3, -20(R4)", 1, 1),
        ("ADD R1, R2, R3", 2, 2), ("ADDI R1, R2, 123", 3, 3),
        ("SUB R4, R5, R6", 4, 4), ("MUL R7, R0, R1", 5, 5),
        ("DIV R2, R3, R4", 6, 6), ("NAND R5, R6, R7", 7, 7),
        ("JMP 1000", 8, 8), ("BEQ R1, R2, -50", 9, 9),
        ("JAL R7, 200", 10, 10), ("RET", 11, 11), # Note: RET parsing is simplified
        ("NOP", 12, 12), ("HALT", 13, 13)
    ]
    for instr_str, addr, uid_val in instructions_to_test:
        try:
            instr = Instruction(instr_str, address=addr, uid=uid_val)
            print(f"Parsed: {instr!r}, FU Type: {instr.get_fu_type()}")
        except ValueError as e:
            print(f"Error parsing '{instr_str}': {e}")

    print("\nTesting invalid instructions:")
    invalid_instructions = [
        ("LD R1, 100(R2)", 100, 100), # Valid, but tests alternative LD name
        ("ADD R1, R2", 101, 101),     # Too few operands
        ("ADDI R1, R2, R3", 102, 102), # R3 instead of Imm
        ("LOAD R8, 0(R0)", 103, 103), # Invalid register R8
        ("STORE R1, (R2)", 104, 104), # Missing offset
        ("UNKNOWN R1, R2, R3", 105, 105), # Unknown op
        ("ADDI R1, R2, 10.5", 106, 106) # Non-integer immediate
    ]
    for instr_str, addr, uid_val in invalid_instructions:
        try:
            instr = Instruction(instr_str, address=addr, uid=uid_val)
            print(f"Parsed (unexpectedly valid): {instr!r}, FU Type: {instr.get_fu_type()}")
        except ValueError as e:
            print(f"Correctly caught error for '{instr_str}': {e}")