import torch
from tqdm import tqdm
from math import sqrt
from utils import BRIMconfig, parse_dimacs_cnf, convert_clauses_to_ising, unsat_count
import matplotlib.pyplot as plt
from typing import Iterable

torch.manual_seed(0) # fix the random seed for reproducibility

def runbrim(P: torch.Tensor, J: torch.Tensor, h: torch.Tensor, clauses: Iterable[Iterable], config: BRIMconfig, spin: torch.Tensor = None, const = 0):
    # Number of annealing steps
    annealing_steps: float = config.t_stop / config.t_step
    h = h.unsqueeze(-1)
    # Number of nodes
    num_nodes = J.shape[0]
    h /= config.R
    J /= config.R
    P /= config.R
    # Random initial spin
    if spin is None:
        spin = 2 * torch.rand(num_nodes, 1) - 1

    # The scale of the noise
    scale_start, scale_end = (config.scale_start, config.scale_end)
    current_scale, scale_step = (
        scale_start,
        (scale_end - scale_start) / annealing_steps,
    )

    unsats = []
    print(len(clauses))
    # Simulation starts
    for i in tqdm(range(int(annealing_steps))):

        # langevin noise
        noise = (
            current_scale
            * torch.randn(num_nodes, 1)
            * sqrt(config.t_step)
            * sqrt(1 / (config.R * config.C))
        )

        # TODO - Implement the "spin dynamics" - You only need one line
        # You can use:
        # spin - current spin vector
        # config.t_step - time step (dt)
        # J - the coupling matrix, it's already divided by Resistance
        # P - the coupling tensor (implementing cubic interactions), it's already divided by Resistance
        # h - linear terms (implementing linear biases), it's already divided by Resistance
        # config.C - the capacitance
        
        # <---- YOUR CODE HERE v---->
        # print(spin.T.matmul(P.matmul(spin)).s)
        gradient = -(J.matmul(spin.sign()) + h + 1/2 * torch.einsum('ijk,jl,kl->il', P, spin.sign(), spin.sign())) * config.t_step / config.C
        spin += gradient
        # <---- YOUR CODE HERE ^---->

        # Add noise
        spin += noise

        # Clip the spin - so the voltage is between -1 and 1
        spin = torch.clip(spin, -1, 1)

        # Update the noisescale
        current_scale += scale_step
        ene = 0

        # save the cut value
        if i % 1000 == 0:
            unsats.append(unsat_count(spin, clauses))
    return unsats


def main():
    config: BRIMconfig = BRIMconfig()
    config.t_stop = 4e-6
    nvars, clauses = parse_dimacs_cnf("uf20-02.cnf")
    h, J, P, const = convert_clauses_to_ising(nvars, clauses)
    cuts = runbrim(h=h, P=P, J=J, clauses=clauses, config=config,const=const)
    print(f"final_unsat_count: {cuts[-1]}, hardware_time: {config.t_stop} seconds")
    fig, ax = plt.subplots()
    ax.set_title("BRIM simulator - UNSAT Count vs. Iteration")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("UNSAT Count")
    ax.set_ylim(0, 50)
    ax.plot(cuts)
    fig.savefig("cuts.png")


if __name__ == "__main__":
    main()