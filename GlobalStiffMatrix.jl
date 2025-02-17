# Arquivo: GlobalStiffMatrix.jl
module GlobalStiffMatrixModule

export GlobalStiffMatrix

using SparseArrays

function GlobalStiffMatrix(str, kv, iK, jK, x, p, elem, emin, mr, strDens)
    # GlobalStiffMatrix constructs the sparse symmetric global stiffness matrix.
    # INPUT:
    #   str - structure with the problem data.
    #   kv - element stiffness matrix, converted into a column vector.
    #   iK - row indexes with nonzero elements of the matrix K.
    #   jK - column indexes with nonzero elements of the matrix K.
    #   x - element densities vector (filtered).
    #   p - penalty parameter of the SIMP model.
    #   elem - structure with element characteristics (deg - polynomial degree, type - element type (1 = Lagrange, 2 = serendipity).
    #   emin - Young's modulus of the void material.
    #   mr - structure that contains parameters used by the multiresolution method.
    #   strDens - structure with density elements grid data for multiresolution.
    # OUTPUT:
    #   K - global stiffness matrix.

    if !mr.op
        sK = kv .* (emin .+ (str.E - emin) .* (x' .^ p))
        sK = vec(sK)
    else
        # Multiresolution
        sK = zeros(size(iK, 1))
        indS = 1
        j = 1
        nely = strDens.nely
        nelxy = strDens.nely * strDens.nelx

        # m = total number of nodes in each element
        m = if elem.type == 1   # Lagrange element
            (elem.deg + 1)^3
        elseif elem.type == 2   # Serendipity element
            8 + 12 * (elem.deg - 1)
        end

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
                v[:, n3+1] = vb .+ n3
            end

            v = sort(vec(v))

            xel = emin .+ (str.E - emin) .* (x[v]' .^ p)
            kel = kv * xel'
            sK[indS:(indS+indstep-1)] = kel
            indS += indstep

            j += mr.n
            if mod(j - 1, nely) == 0
                j += (mr.n - 1) * nely
                if mod(j, nelxy) == 1
                    j += (mr.n - 1) * nelxy
                end
            end
        end
    end

    K = sparse(vec(iK), vec(jK), vec(sK))
    K = K[vec(str.freedofs), vec(str.freedofs)]  # Apply the boundary conditions

    return K
end

end
