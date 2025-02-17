# Arquivo: VCycle.jl
module VCycleModule

export VCycle


# Incluindo o arquivo do módulo
include("Smooth.jl")


# Usando o módulo local
using .SmoothModule # O ponto indica um módulo local
using IterativeSolvers

function VCycle(
    A,
    b,
    x0,
    niter1,
    niter2,
    ngrids,
    nv,
    smoother,
    P,
    M,
    L,
    U,
    R,
    perm,
    opchol,
    opcond,
    pcgp,
)
    # VCycle applies the Multigrid V-Cycle recursively.
    # INPUT:
    #   A - array containing the system matrices on each grid.
    #   b - right hand side vector.
    #   x0 - initial guess.
    #   niter1 - number of pre-smooth iterations.
    #   niter2 - number of post-smooth iterations.
    #   ngrids - number of grids.
    #   nv - grid level.
    #   smoother - smoother (0 - Jacobi, 1 - Gauss-Seidel/SOR, 2 - SSOR).
    #   P - array containing the prolongation matrices.
    #   M - array containing the smoother iteration matrices.
    #   L - array containing the lower triangular smoother iteration matrices (for SSOR).
    #   U - array containing the upper triangular smoother iteration matrices (for SSOR).
    #   R - Cholesky factor of the system matrix on the coarsest grid / or the preconditioner to use PCG on the coarsest grid.
    #   perm - approximate minimum degree permutation vector of the system matrix on the coarsest grid.
    #   opchol - option of what to do on the coarsest grid (0 - increase ngrids, 1 - use PCG).
    #   opcond - Preconditioner option for PCG (0 - diag, 1 - ichol).
    #   pcgp - structure with several parameters used by the PCG algorithm.
    # OUTPUT:
    #   x - approximated solution.


    # Pre-smooth
    if smoother == 2
        x = Smooth(A[nv], M[nv], L[nv], U[nv], b, x0, niter1, smoother)
    else
        x = Smooth(A[nv], M[nv], nothing, nothing, b, x0, niter1, smoother)
    end
    # Compute residual
    rh = b - A[nv] * x

    # Restriction (coarse grid residual)
    rH = P[nv]' * rh

    if nv == ngrids - 1
        # Coarsest grid solve
        if opchol == 1
            if opcond == 1
                eH, _, _ = IterativeSolvers.pcg(
                    A[nv+1][perm, perm],
                    rH[perm],
                    pcgp.tolPCG,
                    pcgp.maxiterPCG,
                    R,
                    R',
                )
                eH[perm] = eH
            else
                eH, _, _ =
                    IterativeSolvers.pcg(A[nv+1], rH, pcgp.tolPCG, pcgp.maxiterPCG, R)
            end
        else
            eH = R \ (R' \ rH[perm])
            eH[perm] = eH
        end
    else
        # Recursively apply V-cycle
        eH = VCycle(
            A,
            rH,
            zeros(length(rH)),
            niter1,
            niter2,
            ngrids,
            nv + 1,
            smoother,
            P,
            M,
            L,
            U,
            R,
            perm,
            opchol,
            opcond,
            pcgp,
        )
    end

    # Prolongation (coarse to fine)
    eh = P[nv] * eH
    x += eh

    # Post-smooth
    if smoother == 2
        x = Smooth(A[nv], M[nv], L[nv], U[nv], b, x, niter2, smoother)
    else
        x = Smooth(A[nv], M[nv], nothing, nothing, b, x, niter2, smoother)
    end
    return x
end


end
