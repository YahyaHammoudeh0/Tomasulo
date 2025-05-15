# femTomas Processor Configuration

# General Configuration
NUM_REGISTERS = 8  # R0-R7, R0 is always 0
MEMORY_SIZE_WORDS = 65536  # 128KB / 2 bytes per word = 65k words
WORD_SIZE_BITS = 16

# Functional Unit Configuration
# You can override this dictionary at runtime using the set_fu_config() function.
# Each entry specifies the number of reservation stations (rs_count) and latency (cycles).
# Latency is the total cycles needed for the operation.

FU_CONFIG = {
    "LOAD":     {"rs_count": 2, "latency": 6},   # 2 (address) + 4 (memory read)
    "STORE":    {"rs_count": 2, "latency": 6},   # 2 (address) + 4 (memory write)
    "BEQ":      {"rs_count": 2, "latency": 1},
    "CALL":     {"rs_count": 1, "latency": 1},   # For CALL/JAL
    "RET":      {"rs_count": 1, "latency": 1},   # For RET
    "ADD_SUB":  {"rs_count": 4, "latency": 2},
    "NOR":      {"rs_count": 2, "latency": 1},
    "MUL":      {"rs_count": 2, "latency": 10},
}

# Pipeline width (for multiple-issue support; not yet implemented in processor)
PIPELINE_WIDTH = 1  # Default single-issue; increase for multiple-issue

def set_fu_config(new_config: dict):
    """
    Override the global FU_CONFIG at runtime.
    Example usage:
        import femTomas.config as config
        config.set_fu_config({ ... })
    """
    global FU_CONFIG
    FU_CONFIG = new_config

def set_pipeline_width(width: int):
    """
    Override the pipeline width (number of instructions issued per cycle).
    """
    global PIPELINE_WIDTH
    PIPELINE_WIDTH = width

# Pipeline Stages (for reference, primary logic driven by FU latencies and RS availability)
# ISSUE_CYCLES = 1
# EXECUTE_CYCLES_VARIES = True # Based on FU_CONFIG
# WRITE_BACK_CYCLES = 1

# Branch Predictor
BRANCH_PREDICTOR_ALWAYS_TAKEN = False # Always predicts "not taken"

# TODO: Potentially add other global simulation parameters here if needed
