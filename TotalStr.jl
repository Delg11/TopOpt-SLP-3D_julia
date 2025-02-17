# Arquivo: TotalStr.jl
module TotalStrModule

export TotalStr

# Incluindo o arquivo do módulo
include("Display3D.jl")

using .Display3DModule  # O ponto indica um módulo local


function TotalStr(str, xstar, sym, prjop)
    # TotalStr generates the complete element densities vector of the optimal structure, based on the symmetries.
    # INPUT: str - structure with the problem data.
    #        xstar - element densities vector (only the symmetric part).
    #        sym - structure with some symmetry options.
    #        prjop - indicates if the densities were rounded to 0 or 1 at the end of the SLP algorithm.
    # OUTPUT: xtotal - complete element densities vector.

    x = xstar

    if sym.xy
        x2 = zeros(str.nelx * str.nely, str.nelz)
        for i = 1:str.nelz
            x2[:, i] = x[(str.nelz-i)*str.nelx*str.nely+1:(str.nelz-i+1)*str.nelx*str.nely]
        end
        x = vcat(vec(x2), x)  # Use `vec(x2)` to flatten `x2` before concatenating
        str.nelz *= 2
    end

    if sym.yz
        n_extra = str.nelx * str.nely - rem(length(x), str.nelx * str.nely)
        aux = reshape(vcat(x, zeros(n_extra)), str.nelx * str.nely, :)
        aux2 = zeros(size(aux))

        i = 1
        i = 1
        for k = str.nelx:-1:1
            for j = 1:str.nely
                idx = (k - 1) * str.nely + j
                if idx <= size(aux, 2)  # Verifica se o índice está dentro do limite de colunas de aux
                    aux2[:, i] = aux[:, idx]
                    i += 1
                else
                    println(
                        "Índice fora dos limites: idx = $idx, size(aux, 2) = $(size(aux, 2))",
                    )
                end
            end
        end

        aux2 = hcat(aux2, aux)'  # Concatenate `aux2` and `aux` horizontally and transpose
        x2 = vec(aux2)  # Flatten `aux2`
        str.nelx *= 2
        str.nx = str.nelx + 1
        x = x2
    end

    xtotal = x

    str.nelem *= (sym.xy + sym.yz + sym.xz) * 2

    Display3D(str, xtotal, prjop)
end


end
