function ElemNeighborhoodWOK(str, rmin, opfilter)
    # ElemNeighborhoodWOK encontra o vizinhança e os fatores de peso de cada elemento finito, para a aplicação do filtro.
    # INPUT: str - estrutura com os dados do problema.
    #        rmin - raio do filtro.
    #        opfilter - opção de filtro (0 = sem filtro, 1 = filtro de densidade média, 2 = filtro de multirresolução).
    # OUTPUT: w - vetor com a soma dos fatores de peso para cada elemento finito.
    # ---------- %

    frx = floor(Int, rmin / str.el)
    fry = floor(Int, rmin / str.eh)
    frz = floor(Int, rmin / str.ew)
    rmin2 = rmin^2

    if opfilter == 1
        alpha = 1 / (2 * ((rmin / 3)^2))
        beta = 1 / (2 * π * (rmin / 3))
    end

    relx = zeros(Int, frz + 1)
    rely = zeros(Int, frz + 1, frx + 1)
    d2 = zeros(Float64, frx + 1, fry + 1, frz + 1)
    w = zeros(Float64, str.nelem)

    for k1 = 1:frz+1
        dz2 = ((k1 - 1) * str.ew)^2
        dx = floor(Int, sqrt(rmin2 - dz2) / str.el)
        relx[k1] = dx
        for i1 = 1:dx+1
            dx2 = ((i1 - 1) * str.el)^2
            dy = floor(Int, sqrt(rmin2 - dz2 - dx2) / str.eh)
            rely[k1, i1] = dy
            if opfilter == 1
                d2[i1, 1:(dy+1), k1] =
                    exp.(-((dz2 .+ dx2 .+ ((0:dy) * str.eh) .^ 2) * alpha)) * beta
            elseif opfilter == 2
                d2[i1, 1:(dy+1), k1] =
                    (rmin - sqrt.(dz2 .+ dx2 .+ ((0:dy) * str.eh) .^ 2)) / rmin
            end

        end
    end

    fix = zeros(Int, str.nelem)
    fix[str.fixedDens] .= 1  # Indica se o elemento tem densidade fixa

    e1 = 0
    for k1 = 1:str.nelz
        for i1 = 1:str.nelx
            for j1 = 1:str.nely
                e1 += 1
                if fix[e1] == 0
                    for k2 = k1:min(k1 + frz, str.nelz)
                        pz = k2 - k1 + 1
                        e21 = (k2 - 1) * str.nelx * str.nely

                        for i2 = max(i1 - relx[pz], 1):min(i1 + relx[pz], str.nelx)
                            px = abs(i2 - i1) + 1
                            e22 = e21 + (i2 - 1) * str.nely
                            minj2 = max(j1 - rely[pz, px], 1)
                            maxj2 = min(j1 + rely[pz, px], str.nely)

                            if minj2 <= (e1 - e22)
                                minj2 = e1 - e22 + 1
                                if maxj2 >= (e1 - e22)
                                    if opfilter == 1
                                        w[e1] += beta
                                    elseif opfilter == 2
                                        w[e1] += 1
                                    end
                                end
                            end

                            for j2 = minj2:maxj2
                                py = abs(j2 - j1) + 1
                                e2 = e22 + j2
                                if fix[e2] == 0
                                    w[e1] += d2[px, py, pz]
                                    w[e2] += d2[px, py, pz]
                                end
                            end
                        end
                    end
                else
                    w[e1] = 1
                end
            end
        end
    end

    return w
end
