# Arquivo: SolveMG.jl
module SolveMGModule

export SolveMG

# Incluindo o arquivo do módulo
include("WCycle.jl")
include("VCycle.jl")


# Usando o módulo local
using .WCycleModule  # O ponto indica um módulo local
using .VCycleModule

using IterativeSolvers
using LinearAlgebra
# x=u[str.freedofs]
# f=str.f
# Ac=Ac
# P=P
# M=M
# L=L
# U=U
# R=R
# perm=perm
# ngrids=mg.ngrids
# cycle=mg.cycle
# smoother=mg.smoother
# smoothiter1=mg.smoothiter1
# smoothiter2=mg.smoothiter2
# tolMG=mg.tolMG
# maxiterMG=mg.maxiterMG
# opchol=mg.opchol
# opcond=mg.opcond
# pcgp=pcgp

function SolveMG(
    x,
    f,
    Ac,
    P,
    M,
    L,
    U,
    R,
    perm,
    ngrids,
    cycle,
    smoother,
    smoothiter1,
    smoothiter2,
    tolMG,
    maxiterMG,
    opchol,
    opcond,
    pcgp,
)
    # SolveMG applies a CG algorithm with multigrid preconditioner to solve the linear system.
    # INPUT: x - initial guess.
    #        f - right hand side vector.
    #        Ac - cell containing the system matrices on each grid.
    #        P - cell containing the prolongation matrices.
    #        M - cell containing the smoother iteration matrices.
    #        L - cell containing the lower triangular smoother iteration matrices (for SSOR).
    #        U - cell containing the upper triangular smoother iteration matrices (for SSOR).
    #        R - Cholesky factor of the system matrix on the coarsest grid / or the preconditioner to use PCG on the coarsest grid.
    #        perm - approximate minimum degree permutation vector of the system matrix on the coarsest grid.
    #        ngrids - number of grids.
    #        cycle - cycle type (0 - Vcycle, 1 - Wcycle, 2 - FullVCycle).
    #        smoother - smoother (0 - Jacobi, 1 - Gauss-Seidel/SOR, 2 - SSOR).
    #        smoothiter1 - number of pre-smooth iterations.
    #        smoothiter2 - number of post-smooth iterations.
    #        tolMG - tolerance for convergence.
    #        maxiterMG - maximum number of iterations.
    #        opchol - option of what to do when the dimension of the system on the coarsest grid is greater than 'nmaxchol' (0 - increase ngrids, 1 - use PCG).
    #        opcond - preconditioner option for PCG to solve the system on the coarsest grid (0 - diag, 1 - ichol).
    #        pcpg - structure that contains several parameters used by the PCG algorithm.
    # OUTPUT: x - approximated solution.
    #         it - number of iterations.
    # ---------- %

    r = f - Ac[1] * vec(x)
    nr = norm(r) / norm(f)  # residual relative norm
    it = 1  # number of iterations
    gamma2 = 0.0
    d = zeros(length(r))
    println("it: $it | nr: $nr")  # Uncomment for debugging output
    while nr > tolMG && it <= maxiterMG
        if cycle == 0
            s = VCycle(
                Ac,
                r,
                zeros(length(r)),
                smoothiter1,
                smoothiter2,
                ngrids,
                1,
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
        elseif cycle == 1
            s = WCycle(
                Ac,
                r,
                zeros(length(r)),
                smoothiter1,
                smoothiter2,
                ngrids,
                1,
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
        elseif cycle == 2
            s = FMVCycle(
                Ac,
                r,
                smoothiter1,
                smoothiter2,
                ngrids,
                1,
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
        rho = dot(r, s)
        if it == 1
            d = s
        else
            beta = rho / gamma2
            d = beta * d + s
        end
        q = Ac[1] * vec(d)
        alpha = rho / dot(d, q)
        x = vec(x) + alpha * d
        r = r - alpha * q
        gamma2 = rho
        nr = norm(r) / norm(f)
        it += 1
        println("it: $it | nr: $nr")  # Uncomment for debugging output
    end

    return x, it
end


end
