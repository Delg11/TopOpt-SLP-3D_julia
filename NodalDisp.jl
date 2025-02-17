# Arquivo: NodalDisp.jl
module NodalDispModule

export NodalDisp

using IterativeSolvers
using LinearAlgebra
using SparseArrays
function NodalDisp(K, f, perm, opsolver, uold, pcgp)
    # NodalDisp calculates the nodal displacements vector.
    # INPUT: 
    #   K - global stiffness matrix.
    #   f - nodal loads vector.
    #   perm - approximate minimum degree permutation vector of K.
    #   opsolver - linear system solver option (0 = Cholesky factorization, 
    #              1 = CG with diagonal preconditioner, 
    #              2 = CG with incomplete Cholesky preconditioner).
    #   uold - previous vector u obtained in the last iteration, initial guess for pcg.
    #   pcgp - structure that contains several parameters for the pcg solver.
    # OUTPUT: 
    #   up - nodal displacements vector (of the free nodes only).
    #   pcgit - number of PCG iterations.
    #   tp - time taken to construct the preconditioner.
    #   ts - time taken to solve the linear system.
    # Initialize output variables
    up = zeros(length(f))
    pcgit = 0
    tp = 0.0
    ts = 0.0

    if opsolver == 0  # Cholesky
        ts = @elapsed begin
            # R = sparse(cholesky(K[perm, perm]).L)'
            # R = sparse(cholesky(K).L)'
            # up[perm] = R \ (R' \ f[perm])
            
            R=cholesky(K)
            up = R \ f
        end
    elseif opsolver == 1  # PCG with diagonal preconditioner
        tp = @elapsed begin
            B = diagm(diag(K))  # Diagonal preconditioner
        end
        ts = @elapsed begin
            if pcgp.opu0
                # In-place CG solver (updates uold directly)
                up = cg!(uold, K, f; abstol = pcgp.tolPCG, maxiter = pcgp.maxiterPCG)
            else
                # Standard CG solver (returns a new solution)
                up = cg(K, f; abstol = pcgp.tolPCG, maxiter = pcgp.maxiterPCG, Pl = B)
            end
        end
    
    elseif opsolver == 2  # PCG with incomplete Cholesky preconditioner
        tp = @elapsed begin
            B = incomplete_cholesky(K[perm, perm]; diagcomp = 0.1)
        end
        ts = @elapsed begin
            up_perm, _, pcgit, _ = pcg(
                K[perm, perm],
                f[perm],
                pcgp.tolPCG,
                pcgp.maxiterPCG,
                Pl = B,
                Pr = B',
                x0 = uold[perm],
            )
            up[perm] = up_perm
        end
    end

    return up, pcgit, tp, ts
end

end
