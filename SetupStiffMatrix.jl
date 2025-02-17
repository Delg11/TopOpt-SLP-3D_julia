# Arquivo: SetupStiffMatrix.jl
module SetupStiffMatrixModule

export SetupStiffMatrix

using SparseArrays
using SuiteSparse
using AMD

function SetupStiffMatrix(str, x, p, elem, Dofs, mr, strDens)
    # SetupStiffMatrix obtains the row and column indexes with nonzero elements of the global stiffness matrix and its permutation vector.
    # INPUT: str - structure with the problem data.
    #        x - initial element densities vector (filtered).
    #        p - penalty parameter of the SIMP model.
    #        elem - structure with element characteristics (deg - polynomial degree, type - element type (1 = Lagrange, 2 = serendipity).
    #        Dofs - matrix with the indexes of the degrees of freedom for each element.
    #        mr - structure that contains parameters used by the multiresolution method.
    #        strDens - structure with density mesh data for multiresolution.
    # OUTPUT: iK - row indexes with nonzero elements of the matrix K.
    #         jK - column indexes with nonzero elements of the matrix K.
    #         perm - approximate minimum degree permutation vector of K.

    # m = total number of nodes in each element
    m = (elem.type == 1) ? (elem.deg + 1)^3 : (8 + 12 * (elem.deg - 1))

    iK = kron(Dofs, ones(3 * m, 1))'
    iK = vec(iK)

    jK = kron(Dofs, ones(1, 3 * m))'
    jK = vec(jK)

    if !mr.op
        sK = (100 * rand((3 * m)^2)) .* (x' .^ p)
        sK = vec(sK)
    else # Multiresolution
        sK = zeros(length(iK))
        indS = 1
        j = 1
        nely = strDens.nely
        nelxy = strDens.nely * strDens.nelx
        indstep = (3 * m)^2

        for i = 1:str.nelem
            vb = zeros(mr.n^2)
            ind = 1
            for n1 = 0:(mr.n-1)
                for n2 = 0:(mr.n-1)
                    vb[ind] = j + n1 * nelxy + n2 * nely
                    ind += 1
                end
            end

            v = zeros(mr.n^2, mr.n)
            for n3 = 0:(mr.n-1)
                v[:, n3+1] .= vb + n3
            end

            v = sort(vec(v))

            xel = x[v] .^ p
            sK[indS:(indS+indstep-1)] .= sum(xel) * (100 * rand(indstep))
            indS += indstep

            j += mr.n
            if (mod(j - 1, nely) == 0)
                j += (mr.n - 1) * nely
                if (mod(j, nelxy)) == 1
                    j += (mr.n - 1) * nelxy
                end
            end
        end
    end

    K = sparse(vec(iK), vec(jK), vec(sK))
    K = K[vec(str.freedofs), vec(str.freedofs)]
    perm = amd(K)  # Obter a permutação

    return iK, jK, perm
end
end
