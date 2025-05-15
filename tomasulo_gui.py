import streamlit as st
import os
from femTomas.processor import Processor

st.set_page_config(page_title="Tomasulo Simulator", layout="wide")
st.title("Tomasulo Algorithm Simulator (Educational GUI)")

# --- Session State Initialization ---
def get_default_fu_config():
    return {
        "LOAD":     {"rs_count": 2, "latency": 6},
        "STORE":    {"rs_count": 2, "latency": 6},
        "BEQ":      {"rs_count": 2, "latency": 1},
        "CALL":     {"rs_count": 1, "latency": 1},
        "RET":      {"rs_count": 1, "latency": 1},
        "ADD_SUB":  {"rs_count": 4, "latency": 2},
        "NOR":      {"rs_count": 2, "latency": 1},
        "MUL":      {"rs_count": 2, "latency": 10},
    }

if 'processor' not in st.session_state:
    st.session_state.processor = None
    st.session_state.program = ''
    st.session_state.fu_config = get_default_fu_config()
    st.session_state.pipeline_width = 1
    st.session_state.cycle = 0
    st.session_state.sim_started = False
    st.session_state.sim_finished = False

# --- Sidebar: Hardware Config ---
st.sidebar.header("Hardware Configuration")
fu_config = {}
for fu, vals in get_default_fu_config().items():
    rs = st.sidebar.number_input(f"{fu} RS Count", min_value=1, value=vals['rs_count'], key=f"rs_{fu}")
    lat = st.sidebar.number_input(f"{fu} Latency", min_value=1, value=vals['latency'], key=f"lat_{fu}")
    fu_config[fu] = {"rs_count": rs, "latency": lat}
pipeline_width = st.sidebar.number_input("Pipeline Width", min_value=1, value=1, key="pipeline_width")

# --- Program Input ---
st.header("1. Load Assembly Program")
prog_source = st.radio("Input Method", ["Paste", "Upload File"])
if prog_source == "Paste":
    program = st.text_area("Paste your assembly program here (labels supported):", height=200)
else:
    uploaded = st.file_uploader("Upload .asm file", type=["asm", "txt"])
    program = uploaded.read().decode() if uploaded else ''

# --- Simulation Controls ---
st.header("2. Simulation Controls")
col1, col2, col3, col4 = st.columns(4)
if col1.button("Initialize/Reset"):
    st.session_state.processor = Processor(fu_config=fu_config, pipeline_width=pipeline_width)
    st.session_state.processor.load_program(program, initial_pc=0)
    st.session_state.cycle = 0
    st.session_state.sim_started = True
    st.session_state.sim_finished = False
    st.success("Simulation initialized.")

if col2.button("Step") and st.session_state.sim_started and not st.session_state.sim_finished:
    st.session_state.processor.run_cycle()
    st.session_state.cycle += 1
    if getattr(st.session_state.processor, 'is_halted', False):
        st.session_state.sim_finished = True

if col3.button("Run to Completion") and st.session_state.sim_started and not st.session_state.sim_finished:
    st.session_state.processor.run_simulation(max_cycles=1000)
    st.session_state.sim_finished = True
    st.session_state.cycle = st.session_state.processor.current_cycle

if col4.button("Reset State"):
    st.session_state.processor = None
    st.session_state.sim_started = False
    st.session_state.sim_finished = False
    st.session_state.cycle = 0
    st.session_state.program = ''

# --- Display State ---
if st.session_state.processor and st.session_state.sim_started:
    st.subheader(f"Cycle: {st.session_state.cycle}")
    st.write("### Reservation Stations")
    rs_data = []
    for fu, rs_list in st.session_state.processor.reservation_stations.items():
        for rs in rs_list:
            rs_data.append({
                "FU Type": fu,
                "Name": getattr(rs, "name", None),
                "Busy": getattr(rs, "busy", None),
                "Op": getattr(getattr(rs, "op_type", None), "name", None),
                "Vj": getattr(rs, "Vj", None),
                "Vk": getattr(rs, "Vk", None),
                "Qj": getattr(rs, "Qj", None),
                "Qk": getattr(rs, "Qk", None),
                "A": getattr(rs, "A", None),
                "Result": rs.get_result() if hasattr(rs, "get_result") else None,
                "Cycles Left": getattr(rs, "remaining_execution_cycles", None),
            })
    st.dataframe(rs_data)

    st.write("### Register File")
    reg_data = []
    for i, val in enumerate(st.session_state.processor.register_file.registers):
        reg_data.append({"Register": f"R{i}", "Value": val, "Tag": st.session_state.processor.register_file.rat[i]})
    st.dataframe(reg_data)

    st.write("### Memory (addresses 0-39)")
    mem_data = []
    memory = st.session_state.processor.memory
    for addr in range(40):
        try:
            value = memory[addr]
        except (TypeError, KeyError, IndexError):
            value = None
        mem_data.append({"Address": addr, "Value": value})
    st.dataframe(mem_data)

    st.write("### Instruction Queue")
    iq_data = []
    for instr in st.session_state.processor.instruction_queue:
        iq_data.append({"Addr": instr.address, "Instruction": str(instr), "Issued": instr.issue_cycle, "ExecStart": instr.execute_start_cycle, "ExecEnd": instr.execute_end_cycle, "WriteBack": instr.write_back_cycle})
    st.dataframe(iq_data)

    if st.session_state.sim_finished:
        st.success("Simulation finished.")
