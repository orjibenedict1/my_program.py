# my_program.py
from perf_takehome import Machine, build_mem_image, Tree, Input, reference_kernel, HASH_STAGES, VLEN


def main():
    tree_height = 10
    batch_size = 256
    rounds = 16

    # Generate tree and input
    tree = Tree.generate(tree_height)
    inp = Input.generate(tree, batch_size=batch_size, rounds=rounds)

    # Build flat memory image
    mem = build_mem_image(tree, inp)

    # Memory offsets
    forest_values_p = mem[4]
    inp_indices_p = mem[5]
    inp_values_p = mem[6]
    n_nodes = mem[1]

    program = []

    # For each round, vectorized by VLEN
    for h in range(rounds):
        for i in range(0, batch_size, VLEN):
            instr = {"load": [], "alu": [],
                     "valu": [], "store": [], "flow": []}

            # Load indices and input values into scratch
            instr["load"].append(
                ("vload", 0, inp_indices_p + i))   # scratch 0..7
            instr["load"].append(
                ("vload", 8, inp_values_p + i))    # scratch 8..15

            # Apply tree traversal and hash per element in vector
            for vi in range(VLEN):
                # XOR with node value
                node_idx_scratch = 0 + vi
                val_scratch = 8 + vi
                instr["load"].append(
                    ("load_offset", 16 + vi, forest_values_p, node_idx_scratch))
                instr["alu"].append(("^", 24 + vi, val_scratch, 16 + vi))

                # Apply all HASH_STAGES
                scratch = 24 + vi
                for stage, (op1, val1, op2, op3, val3) in enumerate(HASH_STAGES):
                    instr["alu"].append(
                        (op1, 32 + stage + vi*len(HASH_STAGES), scratch, scratch))
                    # Constants ignored for now; can expand for full optimization

                # Compute next index based on even/odd
                instr["alu"].append(("%", 100 + vi, 24 + vi, 2))
                instr["alu"].append(("+", 0 + vi, 0 + vi, 100 + vi))
                instr["flow"].append(
                    ("select", 0 + vi, 100 + vi, 0 + vi, 0 + vi))

            # Store back updated input values and indices
            instr["store"].append(("vstore", inp_values_p + i, 24))
            instr["store"].append(("vstore", inp_indices_p + i, 0))

            program.append(instr)

    # Run Machine
    m = Machine(mem, program, debug_info=None)
    m.run()

    # Print first 10 values after Machine
    print("Machine memory values (first 10 inputs):",
          mem[inp_values_p:inp_values_p + 10])

    # Run reference kernel for comparison
    reference_kernel(tree, inp)
    print("Reference kernel values (first 10 inputs):", inp.values[:10])


if __name__ == "__main__":
    main()
