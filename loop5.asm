# Increment R1 five times, then halt. Demonstrates label support.
ADDI R1, R0, 0      # R1 = 0
ADDI R5, R0, 5      # R5 = 5

loop:
    ADDI R1, R1, 1  # R1 = R1 + 1
    BEQ R1, R5, end # If R1 == 5, branch to end
    JMP loop

end:
    HALT
