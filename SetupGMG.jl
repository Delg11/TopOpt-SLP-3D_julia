# Arquivo: SetupGMG.jl
module SetupGMGModule

export SetupGMG

# Incluindo o arquivo do módulo
include("Grad.jl")
include("Prolongation.jl")


# Usando o módulo local
using .GradModule  # O ponto indica um módulo local
using .ProlongationModule

using AMD
using LinearAlgebra
using IncompleteLU
using SparseArrays


function SetupGMG(K, str, mg, elem)
    # SetupGMG performs the setup for geometric multigrid.
    # INPUT:
    #   K - global stiffness matrix.
    #   str - structure with the problem data.
    #   mg - structure that contains several parameters used by the multigrid method.
    #   elem - structure with element characteristics (deg - polynomial degree, type - element type (1 = Lagrange, 2 = serendipity).
    # OUTPUT:
    #   A - array containing the system matrices on each grid.
    #   P - array containing the prolongation matrices.
    #   M - array containing the smoother iteration matrices.
    #   L - array containing the lower triangular smoother iteration matrices (for SSOR).
    #   U - array containing the upper triangular smoother iteration matrices (for SSOR).
    #   R - Cholesky factor of the system matrix on the coarsest grid / or the preconditioner to use PCG on the coarsest grid.
    #   perm - approximate minimum degree permutation vector of the system matrix on the coarsest grid.

    ngrids = mg.ngrids
    smoother = mg.smoother
    omega = mg.omega

    # Grid parameters
    nelx = zeros(Int, ngrids) # Number of elements in the length (x) direction
    nely = zeros(Int, ngrids) # Number of elements in the height (y) direction
    nelz = zeros(Int, ngrids) # Number of elements in the width (z) direction
    nelx[1], nely[1], nelz[1] = str.nelx, str.nely, str.nelz
    for i = 2:ngrids
        nelx[i] = div(nelx[i-1], 2)
        nely[i] = div(nely[i-1], 2)
        nelz[i] = div(nelz[i-1], 2)
    end
    nx = nelx .* elem.deg .+ 1
    ny = nely .* elem.deg .+ 1
    nz = nelz .* elem.deg .+ 1
    nnodes = (nx .* ny) .* nz # Total number of nodes on each grid

    # Operators construction
    freedofs = Vector{Any}(undef, ngrids) # Indexes of the free degrees of freedom on each grid
    freedofs[1] = str.freedofs
    P = Vector{Any}(undef, ngrids - 1)  # Prolongation matrices
    for i = 1:(ngrids-1)
        if elem.type == 1 || elem.deg == 1  # Lagrange element
            P[i], freedofs[i+1] = Prolongation(
                nx[i],
                ny[i],
                nz[i],
                nx[i+1],
                ny[i+1],
                nz[i+1],
                nnodes[i+1],
                freedofs[i],
            )
        elseif elem.type == 2  # Serendipity element
            P[i], freedofs[i+1] = ProlongationS(
                nelx[i],
                nely[i],
                nelz[i],
                nelx[i+1],
                nely[i+1],
                nelz[i+1],
                freedofs[i],
                elem,
            )
        end
    end

    A = Vector{Any}(undef, ngrids)  # System matrices on each grid
    A[1] = K
    for i = 2:ngrids
        A[i] = P[i-1]' * A[i-1] * P[i-1]
    end

    perm = amd(A[ngrids])  # Approximate minimum degree permutation vector of A[ngrids]
    if mg.opchol == 1
        if mg.opcond == 1
            R = ichol(A[ngrids][perm, perm], diagcomp = 0.1)
        else
            R = diagm(diag(A[ngrids]))
        end
    else
        R = sparse(cholesky(Symmetric(A[ngrids][perm, perm])).L)'
    end

    M = Vector{Any}(undef, ngrids - 1)  # Smoother iteration matrices
    L = Vector{Any}(undef, ngrids - 1)  # Lower triangular smoother iteration matrices (for SSOR)
    U = Vector{Any}(undef, ngrids - 1)  # Upper triangular smoother iteration matrices (for SSOR)
    for i = 1:(ngrids-1)
        if smoother == 0  # Jacobi
            M[i] = (1 / omega) * diagm(diag(A[i]))
        elseif smoother == 1  # Gauss-Seidel/SOR
            M[i] = (1 / omega) * diagm(diag(A[i])) + tril(A[i], -1)
        elseif smoother == 2  # SSOR
            L[i] = (1 / omega) * diagm(diag(A[i])) + tril(A[i], -1)
            U[i] = diagm(1.0 ./ diag(A[i])) * L[i]'
            L[i] = (omega / (2 - omega)) * L[i]
        end
    end

    return A, P, M, L, U, R, perm
end
end
