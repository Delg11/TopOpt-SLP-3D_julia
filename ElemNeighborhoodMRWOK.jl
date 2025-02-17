# Arquivo: ElemNeighborhoodMRWOK.jl
module ElemNeighborhoodMRWOKModule

export ElemNeighborhoodMRWOK

function ElemNeighborhoodMRWOK(strDens, strDsgn, rmin, mr, opfilter)
    # ElemNeighborhoodMRWOK finds the neighborhood between the density elements and design variable elements, for multiresolution 
    # INPUT: strDens - structure with density elements grid data for multiresolution. 
    #        strDsgn - structure with design variable elements grid data for multiresolution.
    #        rmin - filter radius. 
    #        mr - structure that contains parameters used by the multiresolution method.
    #        opfilter - filter option (0 = no filter, 1 = mean density filter, 2 = multiresolution filter).
    # OUTPUT: w - vector with the sum of the weight factors for each density element. 
    # ---------- 

    w = zeros(strDens.nelem)
    frx = floor(Int, rmin / strDens.el)
    fry = floor(Int, rmin / strDens.eh)
    frz = floor(Int, rmin / strDens.ew)

    if opfilter == 1
        alpha = 1 / (2 * ((rmin / 3)^2))
        beta = 1 / (2 * π * (rmin / 3))
    end

    rmin2 = rmin^2
    elhDens = strDens.el / 2
    ehhDens = strDens.eh / 2
    ewhDens = strDens.ew / 2
    elhDsgn = strDsgn.el / 2
    ehhDsgn = strDsgn.eh / 2
    ewhDsgn = strDsgn.ew / 2
    nelxyDsgn = strDsgn.nelx * strDsgn.nely
    gridratio = mr.n / mr.d
    ind = 1
    fixDens = zeros(Int, strDens.nelem)
    fixDens[strDens.fixedDens] .= 1
    fixDsgn = zeros(Int, strDsgn.nelem)
    fixDsgn[strDsgn.fixedDens] .= 1
    e1 = 0

    for k1 = 1:strDens.nelz
        z1 = ewhDens + (k1 - 1) * strDens.ew
        mink2 = max(round(Int, max(k1 - frz, 1) / gridratio), 1)
        maxk2 = min(round(Int, min(k1 + frz, strDens.nelz) / gridratio), strDsgn.nelz)
        k2 = mink2:maxk2
        z2 = ewhDsgn + (k2 - 1) * strDsgn.ew
        dz2 = (z2 .- z1) .^ 2

        for i1 = 1:strDens.nelx
            x1 = elhDens + (i1 - 1) * strDens.el
            mini2 = max(round(Int, max(i1 - frx, 1) / gridratio), 1)
            maxi2 = min(round(Int, min(i1 + frx, strDens.nelx) / gridratio), strDsgn.nelx)
            i2 = mini2:maxi2
            x2 = elhDsgn + (i2 - 1) * strDsgn.el
            dx2 = (x2 .- x1) .^ 2

            for j1 = 1:strDens.nely
                e1 += 1
                y1 = ehhDens + (j1 - 1) * strDens.eh
                minj2 = max(round(Int, max(j1 - fry, 1) / gridratio), 1)
                maxj2 =
                    min(round(Int, min(j1 + fry, strDens.nely) / gridratio), strDsgn.nely)
                j2 = minj2:maxj2
                y2 = ehhDsgn + (j2 - 1) * strDsgn.eh
                dy2 = (y2 .- y1) .^ 2

                d2 =
                    repeat(dx2, inner = (length(dy2), 1)) .+
                    repeat(dy2, outer = (1, length(dx2)))
                d2 =
                    repeat(dz2, inner = (length(d2), 1)) .+
                    repeat(d2, outer = (1, length(dz2)))

                e21 = (k2 .- 1) .* nelxyDsgn
                e22 = (i2 .- 1) .* strDsgn.nely
                nb =
                    repeat(e21, inner = (length(e22), 1)) .+
                    repeat(e22, outer = (length(e21), 1))
                nb =
                    repeat(nb, inner = (length(j2), 1)) .+
                    repeat(j2, outer = (length(nb), 1))

                dind = findall(d2 .<= rmin2)
                d = sqrt.(d2[dind])
                ld = ind + length(d)
                nbd = nb[dind]

                for j2 in eachindex(nbd)
                    if (fixDens[e1] == 0 && fixDsgn[nbd[j2]] == 0) ||
                       (fixDens[e1] == 1 && fixDsgn[nbd[j2]] == 1)
                        if opfilter == 1
                            w[e1] += exp.(-(d[j2]^2) * alpha) * beta
                        elseif opfilter == 2
                            w[e1] += (rmin - d[j2]) / rmin
                        end
                    end
                end
                ind = ld
            end
        end
    end

    return w
end
end
