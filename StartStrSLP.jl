# ----------------------------------------------------------------------
# Arquivo: StartStrSLP.jl
# Módulo responsável pela função StartStrSLP
# ----------------------------------------------------------------------

module StartStrSLPModule

# Exporta a função StartStrSLP para uso externo
export StartStrSLP

# Importa pacotes e bibliotecas necessários
using Dates
using DataFrames
using AMDGPU

# Inclui o arquivo do módulo StartMR
include("StrSLP.jl")
include("StartMR.jl")
include("ElemNeighborhoodMR.jl")
include("ElemNeighborhood.jl")
include("ElemStiffMatrixGQMR.jl")
include("ElemStiffMatrixGQ.jl")
include("ElemStiffMatrix.jl")
include("ElemNeighborhoodMRWOK.jl")
include("DegreesOfFreedom.jl")
include("ApplyFilter.jl")
include("SetupStiffMatrix.jl")

# Importa o módulo local StartMR
using .StrSLPModule  # O ponto indica um módulo local
using .StartMRModule  # O ponto indica um módulo local
using .ElemNeighborhoodMRModule
using .ElemNeighborhoodModule
using .ElemStiffMatrixGQMRModule
using .ElemStiffMatrixGQModule
using .ElemStiffMatrixModule
using .ElemNeighborhoodMRWOKModule
using .DegreesOfFreedomModule
using .ApplyFilterModule
using .SetupStiffMatrixModule

mutable struct Timing
    prefilter::Float64  # Tempo para obter vizinhos e fatores de peso
    filter::Float64     # Tempo para aplicar o filtro
    setupK::Float64     # Tempo para construir as matrizes de rigidez
    setupPrec::Float64   # Tempo para configurar o pré-condicionador
    system::Float64     # Tempo para resolver os sistemas lineares
    grad::Float64       # Tempo para calcular o gradiente da função objetivo
    LP::Float64         # Tempo para resolver os subproblemas LP
    fix::Float64        # Tempo para escolher e fixar elementos
    proj::Float64       # Tempo para arredondar densidades para 0 ou 1
    other::Float64
    total::Float64
end

mutable struct TempData
    heavi::Union{Bool,Nothing}          # Tipo logical (equivalente a Bool em Julia)
    fix::Union{Int,Nothing}             # Tipo int (escalares inteiros)
    prjop::Union{Bool,Nothing}          # Tipo logical (equivalente a Bool em Julia)
    tolG::Union{Float64,Nothing}        # Tipo float (escalares flutuantes)
    time::Union{DataFrame,Nothing}  # Tipo table (usando Array 2x11 genérico, pois "table" não tem um equivalente direto)
    rownames::Union{Vector{String},Nothing}  # Tipo string (vetor de strings)
    iter::Union{Int,Nothing}            # Tipo int (escalares inteiros)
    itrej::Union{Int,Nothing}           # Tipo int (escalares inteiros)
    ns::Union{Int,Nothing}              # Tipo int (escalares inteiros)
    pcgit::Union{Vector{Int},Nothing}   # Tipo int (vetor de inteiros 48x1)
    lpit::Union{Vector{Int},Nothing}    # Tipo int (vetor de inteiros 47x1)
end

# Definição de um Timer mutável
mutable struct Timer
    elapsed_time::Float64  # Tempo total acumulado
    start_time::Float64    # Tempo de início para a medida atual
    active::Bool           # Indica se o timer está ativo

    function Timer()
        new(0.0, 0.0, false)
    end
end

function tic(timer::Timer)
    timer.start_time = time()   # Captura o tempo atual
    timer.active = true          # Indica que o timer está ativo
    return timer.start_time 
end

function toc(timer::Timer)
    if timer.active
        elapsed = time() - timer.start_time  # Calcula o tempo decorrido
        timer.elapsed_time += elapsed         # Acumula o tempo total
        return elapsed                        # Retorna o tempo decorrido
    else
        error("Timer is not active. Please call 'tic' before 'toc'.")
    end
end

function get_elapsed(timer::Timer)
    return timer.elapsed_time  # Método para acessar o tempo total acumulado
end

mutable struct MaterialDsgn
    l::Union{Float64,Nothing}
    h::Union{Float64,Nothing}
    w::Union{Float64,Nothing}
    nelx::Union{Int,Nothing}
    nely::Union{Int,Nothing}
    nelz::Union{Int,Nothing}
    E::Union{Float64,Nothing}
    nu::Union{Float64,Nothing}
    el::Union{Float64,Nothing}
    eh::Union{Float64,Nothing}
    ew::Union{Float64,Nothing}
    nx::Union{Int,Nothing}
    ny::Union{Int,Nothing}
    nz::Union{Int,Nothing}
    nelem::Union{Int,Nothing}
    nnodes::Union{Int,Nothing}
    fdv::Union{Matrix{Float64},Nothing}
    freeDens::Union{Matrix{Int},Nothing}
    fixedDens::Union{Matrix{Int},Nothing}
    fixedDensVal::Union{Matrix{Float64},Nothing}
    freeDensG::Union{Matrix{Int},Nothing}   # Elements that will be in the computation of the gradient
end

function StartStrSLP(
    str,
    volfrac,
    p,
    p123,
    emin,
    opfilter,
    rmin,
    volineq,
    elem,
    slp,
    opsolver,
    pcgp,
    mg,
    prj,
    genK,
    mr,
)
    # StartStrSLP initializes the SLP algorithm to solve the topology optimization problem. #
    # INPUT: str - structure with the problem data.
    #        volfrac - maximum volume fraction of the domain that the structure can occupy.
    #        p - penalty parameter for the SIMP model. 
    #        p123 - indicates if the continuation strategy (use p = 1,2,3) will be adopted. 
    #        emin - Young's modulus of the void material.
    #        opfilter - filter option (0 = no filter, 1 = weighted average density filter, 2 = average density filter).
    #        rmin - filter radius.
    #        volineq - type of the volume constraint (true = "less than or equal to", false = "equal to").
    #        elem - structure with element characteristics (deg - polynomial degree, type - element type (1 = Lagrange, 2 = serendipity).
    #        slp - structure that contains several parameters used by the SLP algorithm.
    #        opsolver - linear system solver option.
    #        pcgp - structure that contains several parameters used by the pcg system solver.
    #        mg - structure that contains several parameters used by the multigrid method.
    #        prj - structure that contains the parameters used for rounding the optimal densities to 0 or 1.
    #        genK - explicit generation of matrix K for multigrid solvers (true = yes, false = no).
    #        mr - structure that contains parameters used by the multiresolution method.
    # OUTPUT: xstar - approximated optimal element densities vector. 
    #         xstarDsgn - approximated optimal design variables vector.
    #         u - approximated nodal displacements vector. 
    #         opstop - stopping criterion achieved.
    #         iter - number of iterations.
    #         itrej - number of rejected steps.
    #         F - objective function optimal value.
    #         vol - volume fraction at the optimal solution.
    #         ns - number of linear systems solved.
    #         pcgit - number of iterations of PCG for each system solved.
    #         lpit - number of iterations of linprog for each LP solved.
    #         time - structure with the time consumed on each part of the program.
    #         xhist - densities of each iteration of the SLP algorithm.
    #         strDens - structure with density mesh data for multiresolution. 
    # ---------- #
    # Definição da estrutura para armazenar os tempos



    println(
        "Solving the topology optimization problem using a Sequential Linear Programming algorithm.",
    )
    timer = Timer()  # Inicializa o timer
    startTime = tic(timer)

    tprefilter = 0.0 # Time consumed to obtain the element neighbors and weight factors
    tfilter = 0.0 # Time consumed to apply the filter
    tsetupK = 0.0 # Time consumed to construct the stiffness matrices 
    tsetupPrec = 0.0 # Time consumed to setup the preconditioner to solve the linear systems
    tsystem = 0.0 # Time consumed to solve the linear systems
    tgrad = 0.0 # Time consumed to calculate the gradient of the objective function
    tLP = 0.0 # Time consumed to solve the LP subproblems 
    tfix = 0.0 # Time consumed to choose and fix elements
    tproj = 0.0 # Time consumed to round the densities to 0 or 1 
    tother = 0.0
    ttotal = 0.0
    t = Timing(
        tprefilter,
        tfilter,
        tsetupK,
        tsetupPrec,
        tsystem,
        tgrad,
        tLP,
        tfix,
        tproj,
        tother,
        ttotal,
    )

    # Checking if there are void or solid regions in the domain %
    if str.fixedDens === nothing
        str.freeDens = collect(1:str.nelem)
        str.fixedDens = []
        str.fixedDensVal = []
    end
    temp = TempData(
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    strDsgn = MaterialDsgn(
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
        nothing,
    )
    # Getting the initial design variables and meshes data for multiresolution
    if mr.op == false
        strDens = nothing
        strDsgn = nothing
        x0 = volfrac * ones(Float64, str.nelem)  # Cria um array de Float64 com valor volfrac
        if str.fixedDens !== nothing && !isempty(str.fixedDens)  # Verifica se fixedDens não é nulo e não está vazio
            x0[str.fixedDens] .= str.fixedDensVal  # Atribui os valores fixos de densidade
            x0[str.freeDens] .=
                (
                    volfrac * str.l * str.h * str.w -
                    sum(str.fixedDensVal) * str.el * str.eh * str.ew
                ) / length(str.freeDens)
        end
    else
        strDens, strDsgn, x0 = StartMR(
            str,
            volfrac,
            p,
            p123,
            emin,
            opfilter,
            rmin,
            volineq,
            elem,
            slp,
            opsolver,
            pcgp,
            mg,
            prj,
            genK,
            mr,
        )
    end
    # ----------
    # Calculating the matrix to interpolate the displacements
    if mr.op && mr.interp
        mr.Idisp, mr.Ldofs, mr.kd = InterpolateDisp(mr, elem, strDens)
    end
    # ----------
    # Adapting the number of nodes, freedofs, vector of nodal loads and multigrid parameters according to the polynomial degree of the shape functions
    if elem.deg > 1
        str = ConvertDofs(str, elem)
        if genK
            mg.smoother = 2
            mg.omega = 1.2
        else
            mg.smoother = 0
            mg.omega = 0.3
        end
    end
    # ---------- #

    # Disabling the incomplete Cholesky preconditioner and warning that genK = false can only be used with opsolver = 1, 3 or 4
    if !genK
        if opsolver == 4 || opsolver == 3
            if mg.smoother != 0
                println(
                    "Using Multigrid with genK = false. Smoother will be set to 0 (Jacobi).",
                )
                mg.smoother = 0
            end
            if opsolver == 3 && mg.opcond != 0
                println(
                    "Using Algebraic Multigrid with genK = false. The preconditioner for test space generation will be set to 0 (diagonal).",
                )
                mg.opcond = 1
            end
        elseif opsolver != 1
            println("opsolver = $(opsolver). genK will be set to true.")
            genK = true
        end
    end
    # ---------- #
    # Checking if the geometric multigrid can be used (if not, try to reduce ngrids or change to algebraic multigrid)
    if opsolver == 4 &&
       (
        mod(str.nelx, 2^(mg.ngrids - 1)) +
        mod(str.nely, 2^(mg.ngrids - 1)) +
        mod(str.nelz, 2^(mg.ngrids - 1))
    ) != 0
        ngridstemp = mg.ngrids
        while (
            mod(str.nelx, 2^(mg.ngrids - 1)) +
            mod(str.nely, 2^(mg.ngrids - 1)) +
            mod(str.nelz, 2^(mg.ngrids - 1))
        ) != 0
            mg.ngrids -= 1
        end
        if mg.ngrids >= 2
            println(
                "You cannot use geometric multigrid in this problem with $ngridstemp grids. ngrids was reduced to $(mg.ngrids).",
            )
        else
            println(
                "You cannot use geometric multigrid in this problem. Using algebraic multigrid instead.",
            )
            opsolver = 3
            mg.ngrids = ngridstemp
        end
    end
    # ---------- #
    # Adjusting ngrids and opchol according to nmaxchol (for geometric multigrid) %
    if (opsolver == 4) && (mg.opchol == 0)
        while (
            (
                (
                    (
                        (str.nelx / (2^(mg.ngrids - 1)) + 1) *
                        (str.nely / (2^(mg.ngrids - 1)) + 1) *
                        (str.nelz / (2^(mg.ngrids - 1)) + 1)
                    ) * 3
                ) > mg.nmaxchol
            ) && (mg.opchol == 0)
        )
            if (
                mod(str.nelx, 2^mg.ngrids) +
                mod(str.nely, 2^mg.ngrids) +
                mod(str.nelz, 2^mg.ngrids)
            ) == 0
                mg.ngrids += 1
                println(
                    "The size of the system for the coarsest grid is greater than nmaxchol. ngrids was increased to $(mg.ngrids)",
                )
            else
                mg.opchol = 1
                println(
                    "The size of the system for the coarsest grid is greater than nmaxchol. PCG will be used.",
                )
            end
        end
    end
    # Element neighbors and weight factors (for the filter application) %

    if opfilter != 0
        tic(timer)

        if genK
            if mr.op  # Multiresolution
                if mr.n != mr.d  # Multiresolution with 3 distinct meshes
                    W, w = ElemNeighborhoodMR(strDens, strDsgn, rmin, mr, opfilter)
                else
                    W, w = ElemNeighborhood(strDsgn, rmin, opfilter)
                end
            else
                W, w = ElemNeighborhood(str, rmin, opfilter)
            end
        else
            if mr.op
                if mr.n != mr.d
                    w = ElemNeighborhoodMRWOK(strDens, strDsgn, rmin, mr, opfilter)
                else
                    w = ElemNeighborhoodWOK(strDsgn, rmin, opfilter)
                end
            else
                w = ElemNeighborhoodWOK(str, rmin, opfilter)
            end
            W = []
        end
        t.prefilter += toc(timer)
    else
        W = []
        w = []
    end

    # Setting up finite element analysis #
    tic(timer)
    if mr.op  # Multiresolution
        k, kv = ElemStiffMatrixGQMR(str, mr.n, elem)  # Stiffness integrands for each density element inside a finite element
    else
        if elem.deg > 1
            k = ElemStiffMatrixGQ(str, elem.deg + 1, elem)  # Element stiffness matrix with Gaussian quadrature
        else
            k = ElemStiffMatrix(str)  # Element stiffness matrix
        end
        kv = vec(k)  # Element stiffness matrix converted into a vector
    end

    if genK
        Dofs = DegreesOfFreedom(str, elem)
        xfil0 = ApplyFilter(Array(x0), W, w, opfilter, false)
        iK, jK, perm = SetupStiffMatrix(str, xfil0, p, elem, Dofs, mr, strDens)
    else
        iK = []  # Inicializa iK como vazio
        jK = []  # Inicializa jK como vazio
        perm = []  # Inicializa perm como vazio
        Dofs = []  # Inicializa Dofs como vazio
        xfil0 = ApplyFilterWOK(x0, str, rmin, w, opfilter, false, mr, strDens, strDsgn)
    end

    t.setupK += toc(timer)
    # ---------- #
    # Disabling some parameters in the first call to StrSLP
    if !prj.op
        temp.heavi = false # Disabling the Heaviside projection if the densities are not to be rounded to 0 or 1.
    else
        temp.heavi = prj.heavi # Storing prj.heavi.
    end
    prj.heavi = false # Disabling the Heaviside projection in the first call to StrSLP.
    temp.fix = elem.fix # Storing elem.fix.
    elem.fix = 0 # Don't fix variables in the first call to StrSLP.
    temp.prjop = prj.op # Storing prj.op
    temp.tolG = slp.tolG # Storing slp.tolG
    if elem.increasedeg
        prj.op = false # Don't round densities in the first call to StrSLP if we are increasing the element degree.
        slp.tolG *= 10 # Increase the minimum value for the projected gradient norm in the first call to StrSLP if we are increasing the element degree.
    end
    # ---------- #

    # Applying the SLP algorithm to solve the problem
    if !p123
        xstar, xstarDsgn, u, opstop, iter, itrej, F, vol, ns, pcgit, lpit, t, xhist =
            StrSLP(
                str,
                volfrac,
                p,
                emin,
                opfilter,
                rmin,
                volineq,
                elem,
                slp,
                opsolver,
                pcgp,
                mg,
                prj,
                genK,
                mr,
                strDens,
                strDsgn,
                x0,
                xfil0,
                W,
                w,
                k,
                kv,
                iK,
                jK,
                perm,
                Dofs,
                t,
            )
    else
        tmp = slp.maxiter
        slp.maxiter = slp.maxit12
        tprj = prj.op
        prj.op = false
        println("Solving the problem with SIMP parameter p = 1.")
        xstar, xstarDsgn, _, _, iter1, itrej1, _, _, ns1, pcgit1, lpit1, t, _ = StrSLP(
            str,
            volfrac,
            1,
            emin,
            opfilter,
            rmin,
            volineq,
            elem,
            slp,
            opsolver,
            pcgp,
            mg,
            prj,
            genK,
            mr,
            strDens,
            strDsgn,
            x0,
            xfil0,
            W,
            w,
            k,
            kv,
            iK,
            jK,
            perm,
            Dofs,
            t,
        )
        println("Solving the problem with SIMP parameter p = 2.")
        xstar, xstarDsgn, _, _, iter2, itrej2, _, _, ns2, pcgit2, lpit2, t, _ = StrSLP(
            str,
            volfrac,
            2,
            emin,
            opfilter,
            rmin,
            volineq,
            elem,
            slp,
            opsolver,
            pcgp,
            mg,
            prj,
            genK,
            mr,
            strDens,
            strDsgn,
            xstarDsgn,
            xstar,
            W,
            w,
            k,
            kv,
            iK,
            jK,
            perm,
            Dofs,
            t,
        )
        slp.maxiter = tmp
        prj.op = tprj
        println("Solving the problem with SIMP parameter p = 3.")
        xstar, xstarDsgn, u, opstop, iter3, itrej3, F, vol, ns3, pcgit3, lpit3, t, xhist =
            StrSLP(
                str,
                volfrac,
                3,
                emin,
                opfilter,
                rmin,
                volineq,
                elem,
                slp,
                opsolver,
                pcgp,
                mg,
                prj,
                genK,
                mr,
                strDens,
                strDsgn,
                xstarDsgn,
                xstar,
                W,
                w,
                k,
                kv,
                iK,
                jK,
                perm,
                Dofs,
                t,
            )
        iter = [iter1, iter2, iter3]
        itrej = [itrej1, itrej2, itrej3]
        ns = [ns1, ns2, ns3]
        pcgit = vcat(pcgit1, pcgit2, pcgit3)
        lpit = vcat(lpit1, lpit2, lpit3)
    end
    # ---------- #

    # Calculating the time consumed
    total = toc(timer) # Total time
    t.other = total - (
        t.prefilter +
        t.filter +
        t.setupK +
        t.setupPrec +
        t.system +
        t.grad +
        t.LP +
        t.fix +
        t.proj
    )
    t.total = total
    
   
    # Creating a DataFrame from the fields of t
    # Create a DataFrame using the values of 't'
    temp.time = DataFrame(
        prefilter = [t.prefilter],
        filter = [t.filter],
        setupK = [t.setupK],
        setupPrec = [t.setupPrec],
        system = [t.system],
        grad = [t.grad],
        LP = [t.LP],
        fix = [t.fix],
        proj = [t.proj],
        other = [t.other],
        total = [t.total]
    )
    # Setting the row name
    temp.rownames = ["First solution"]
    

    # # Storing the number of iterations
    temp.iter = iter
    temp.itrej = itrej
    temp.ns = ns
    temp.pcgit = pcgit
    temp.lpit = lpit
    # ---------- #

    # Solve the problem again if we are increasing the element degree
    if elem.increasedeg
        if genK
            # Change smoother parameters to improve multigrid with element degree greater than 1
            mg.smoother = 2
            mg.omega = 1.2
        else
            mg.smoother = 0
            mg.omega = 0.3
        end
        while elem.deg < elem.maxdeg
            # Restart the timer
            startTime = time()
            t.prefilter = 0.0
            t.filter = 0.0
            t.setupK = 0.0
            t.setupPrec = 0.0
            t.system = 0.0
            t.grad = 0.0
            t.LP = 0.0
            t.fix = 0.0
            t.proj = 0.0
            t.other = 0.0
            t.total = 0.0
            
            elem.fix = temp.fix
            x0 = xstarDsgn
            xfil0 = xstar
            elem.deg += 1
            oldstr = str

            # Use serendipity elements if the degree is greater than 2
            if elem.deg > 2
                elem.type = 2 
            end
            # Recover the defined minimum value for the projected gradient norm if the element degree is maximum
            if elem.deg == elem.maxdeg
                slp.tolG = temp.tolG
                prj.op = temp.prjop
            end

            # Adapting the number of nodes, freedofs and vector of nodal loads according to the polynomial degree of the shape functions
            str = ConvertDofs(str, elem)

            # Saving the original freedofs and vector of nodal loads if we will choose variables to be fixed
            if elem.fix != 0
                str.oldfreedofs = str.freedofs
                str.suppindex = setdiff(1:3*str.nnodes, str.oldfreedofs)
                str.oldf = zeros(3*str.nnodes)
                str.oldf[str.oldfreedofs] = str.f
            end

            # Recalculating the matrix to interpolate the displacements
            if mr.op && mr.interp
                mr.Idisp, mr.Ldofs, mr.kd = InterpolateDisp(mr, elem, strDens)
            end

            # Setting up finite element analysis
            startTime = time()
            if mr.op # Multiresolution
                k, kv = ElemStiffMatrixGQMR(str, mr.n, elem) # Stiffness integrands for each density element inside a finite element
            else
                if elem.deg > 1
                    k = ElemStiffMatrixGQ(str, elem.deg + 1, elem) # Element stiffness matrix with Gaussian quadrature
                else
                    k = ElemStiffMatrix(str) # Element stiffness matrix
                end
                kv = vec(k) # Element stiffness matrix converted into a vector
            end
            if genK
                Dofs = DegreesOfFreedom(str, elem)
                iK, jK, perm = SetupStiffMatrix(str, xstar, p, elem, Dofs, mr, strDens)
            end
            t[:setupK] += time() - startTime

            # Calling StrSLP
            println("\nSolving the problem with element degree equal to $(elem.deg)")
            xstar, xstarDsgn, u, opstop, iter4, itrej4, F, vol, ns4, pcgit4, lpit4, t, xhist = 
                StrSLP(str, volfrac, p, emin, opfilter, rmin, volineq, elem, slp, opsolver, pcgp, mg, prj, genK, mr, strDens, strDsgn, x0, xfil0, W, w, k, kv, iK, jK, perm, Dofs, t)

            # Storing the number of iterations
            push!(temp.iter, iter4)
            push!(temp.itrej, itrej4)
            push!(temp.ns, ns4)
            push!(temp.pcgit, pcgit4)
            push!(temp.lpit, lpit4)

            # Calculating the time consumed
            total = time() - startTime # Total time
            t[:other] = total - (t[:prefilter] + t[:filter] + t[:setupK] + t[:setupPrec] + t[:system] + t[:grad] + t[:LP] + t[:fix] + t[:proj])
            t[:total] = total
            temp.time = vcat(temp.time, DataFrame(t))
            temp.rownames = vcat(temp.rownames, "Solution w/ deg " * string(elem.deg))
        end
    end
    # ----------
    # Solve the problem again when rounding the densities, until finding an optimal solution
    prj.op = temp.prjop
    if prj.op
        # Restart the timer
        startTime = time()
        t.prefilter = 0.0
        t.filter = 0.0
        t.setupK = 0.0
        t.setupPrec = 0.0
        t.system = 0.0
        t.grad = 0.0
        t.LP = 0.0
        t.fix = 0.0
        t.proj = 0.0
        t.other = 0.0
        t.total = 0.0
        # Restoring some values
        prj.heavi = temp.heavi
        opfilter = prj.opfilter

        # Saving the original freedofs and vector of nodal loads if we will choose elements to be fixed
        if elem.fix != 0
            str.oldfreedofs = str.freedofs
            str.suppindex = setdiff(1:3*str.nnodes, str.oldfreedofs)
            str.oldf = zeros(3*str.nnodes)
            str.oldf[str.oldfreedofs] = str.f
        end

        # Use linear elements if prj.deg = true
        if elem.deg > 1 && prj.deg
            elem.deg = 1
            str = oldstr

            # Recalculating the matrix to interpolate the displacements
            if mr.op && mr.interp
                mr.Idisp, mr.Ldofs = InterpolateDisp(mr, elem)
            end

            # Setting up finite element analysis
            start_time = time()
            if mr.op # Multiresolution
                k, kv = ElemStiffMatrixGQMR(str, mr.n, elem) # Stiffness integrands for each density element inside a finite element
            else
                if elem.deg > 1
                    k = ElemStiffMatrixGQ(str, elem.deg + 1, elem) # Element stiffness matrix with Gaussian quadrature
                else
                    k = ElemStiffMatrix(str) # Element stiffness matrix
                end
                kv = vec(k) # Element stiffness matrix converted into a vector
            end
            if genK
                Dofs = DegreesOfFreedom(str, elem)
                iK, jK, perm = SetupStiffMatrix(str, xstar, p, elem, Dofs, mr, strDens)
            end
            t.setupK += time() - start_time
        end

        # Recomputing element neighbors and weight factors (for the filter application)
        if opfilter != 0
            if prj.rmin < rmin && !elem.increasedeg
                rmin = prj.rmin
                start_time = time()
                if genK
                    if mr.op # Multiresolution
                        if mr.n != mr.d # Multiresolution with 3 distinct meshes
                            W, w = ElemNeighborhoodMR(strDens, strDsgn, rmin, mr, opfilter)
                        else
                            W, w = ElemNeighborhood(strDsgn, rmin, opfilter)
                        end
                    else
                        W, w = ElemNeighborhood(str, rmin, opfilter)
                    end
                else
                    if mr.op
                        if mr.n != mr.d
                            w = ElemNeighborhoodMRWOK(strDens, strDsgn, rmin, mr, opfilter)
                        else
                            w = ElemNeighborhoodWOK(strDsgn, rmin, opfilter)
                        end
                    else
                        w = ElemNeighborhoodWOK(str, rmin, opfilter)
                    end
                    W = []
                end
                t.prefilter += time() - start_time
            end
        else
            W, w = [], []
        end

        start_time = time()
        if prj.heavi
            velem = ((str.l * str.h * str.w) / (mr.op ? strDens.nelem : str.nelem)) * ones(str.nelem)
            vcur = dot(velem, xstar)
            vdes = str.l * str.h * str.w * volfrac
            xstar, prj = HeavisideProj(xstar, prj, velem, vcur, vdes, true)
        end

        rndfrac = round(length(xstar) * volfrac) / length(xstar)
        t.proj += time() - start_time
        eleq0 = count(x -> x <= 0, xstar)
        eleq1 = count(x -> x >= 1, xstar)
        elint = length(xstar) - eleq0 - eleq1
        println("Void elements: $eleq0.  Full elements: $eleq1. Intermediate elements: $elint. Volume fraction: $(vol * 100)")

        # Calling StrSLP
        itproj = 0
        xold = zeros(length(xstar))
        while itproj < prj.maxit && ((norm(xold - xstar, 1) > norm(xold, 1) * prj.tolN) || ((vol - rndfrac) > prj.tolV))
            xold = xstar
            x0 = (mr.op && mr.n != mr.d) ? xstarDsgn : xold

            start_time = time()
            xfil0 = genK ? ApplyFilter(x0, W, w, opfilter, false) : ApplyFilterWOK(x0, str, rmin, w, opfilter, false, mr, strDens, strDsgn)
            t.filter += time() - start_time

            println("\nAttempt $(itproj + 1) to solve the problem after rounding the densities to 0 or 1 using the gradient of the Lagrangian.")
            xstar, xstarDsgn, uold, opstop, iter4, itrej4, _, vol, ns4, pcgit4, lpit4, t, _ = StrSLP(str, volfrac, p, emin, opfilter, rmin, volineq, elem, slp, opsolver, pcgp, mg, prj, genK, mr, strDens, strDsgn, x0, xfil0, W, w, k, kv, iK, jK, perm, Dofs, t)

            if prj.heavi
                start_time = time()
                xstar, prj = HeavisideProj(xstar, prj, velem, dot(velem, xstar), vdes, true)
                t.proj += time() - start_time
            end
            eleq0 = count(x -> x <= 0, xstar)
            eleq1 = count(x -> x >= 1, xstar)
            elint = length(xstar) - eleq0 - eleq1
            println("Void elements: $eleq0.  Full elements: $eleq1. Intermediate elements: $elint. Volume fraction: $(vol * 100)")
            itproj += 1

            # Storing the number of iterations
            push!(temp.iter, iter4)
            push!(temp.itrej, itrej4)
            push!(temp.ns, ns4)
            append!(temp.pcgit, pcgit4)
            append!(temp.lpit, lpit4)
        end

        # Calculating the time consumed
        total = time() - startTime
        t.other = total - (t.prefilter + t.filter + t.setupK + t.setupPrec + t.system + t.grad + t.LP + t.fix + t.proj)
        t.total = total
        push!(temp.time, DataFrame(t))
        push!(temp.rownames, "Round densities")

        # Calculate the objective function value of the final solution with rounded densities
        u = zeros(3 * str.nnodes)
        if opsolver == 3 # Using AMG
            if genK
                Ac = GlobalStiffMatrix(str, kv, iK, jK, xstar, p, elem, emin, mr, strDens)
                Ac, P, M, L, U, R, perm, mg.ngrids = SetupAMG(Ac, mg)
                u[str.freedofs] = SolveMG(uold[str.freedofs], str.f, Ac, P, M, L, U, R, perm, mg.ngrids, mg.cycle, mg.smoother, mg.smoothiter1, mg.smoothiter2, mg.tolMG, mg.maxiterMG, mg.opchol, mg.opcond, pcgp)
            else
                An, P, M, R, perm, mg.ngrids = SetupAMGWOK(str, mg, kv, xstar, p, emin, elem, mr)
                u[str.freedofs] = SolveMGWOK(uold[str.freedofs], An, P, M, R, perm, mg.ngrids, mg.cycle, mg.smoother, mg.smoothiter1, mg.smoothiter2, mg.tolMG, mg.maxiterMG, mg.opchol, mg.opcond, kv, xstar, p, emin, pcgp, str, elem, mr)
            end
        elseif opsolver == 4 # Using GMG
            if genK
                Ac = GlobalStiffMatrix(str, kv, iK, jK, xstar, p, elem, emin, mr, strDens)
                Ac, P, M, L, U, R, perm = SetupGMG(Ac, str, mg, elem)
                u[str.freedofs] = SolveMG(uold[str.freedofs], str.f, Ac, P, M, L, U, R, perm, mg.ngrids, mg.cycle, mg.smoother, mg.smoothiter1, mg.smoothiter2, mg.tolMG, mg.maxiterMG, mg.opchol, mg.opcond, pcgp)
            else
                An, P, M, R, perm = SetupGMGWOK(str, mg, kv, xstar, p, emin, elem, mr)
                u[str.freedofs] = SolveMGWOK(uold[str.freedofs], An, P, M, R, perm, mg.ngrids, mg.cycle, mg.smoother, mg.smoothiter1, mg.smoothiter2, mg.tolMG, mg.maxiterMG, mg.opchol, mg.opcond, kv, xstar, p, emin, pcgp, str, elem, mr)
            end
        elseif opsolver == 1 && !genK
            u[str.freedofs] = NodalDispWOK(str, kv, xstar, p, emin, uold[str.freedofs], pcgp, elem, mr)
        else
            K = GlobalStiffMatrix(str, kv, iK, jK, xstar, p, elem, emin, mr, strDens)
            u[str.freedofs] = NodalDisp(K, str.f, perm, opsolver, uold[str.freedofs], pcgp)
        end
        F = dot(str.f, u[str.freedofs])
    end

    # Calculating the time consumed
    total = toc(timer) # Total time
    t.other = total - (
        t.prefilter +
        t.filter +
        t.setupK +
        t.setupPrec +
        t.system +
        t.grad +
        t.LP +
        t.fix +
        t.proj
    )
    t.total = total
    
   
    # Creating a DataFrame from the fields of t
    # Create a DataFrame using the values of 't'
    temp.time = DataFrame(
        prefilter = [t.prefilter],
        filter = [t.filter],
        setupK = [t.setupK],
        setupPrec = [t.setupPrec],
        system = [t.system],
        grad = [t.grad],
        LP = [t.LP],
        fix = [t.fix],
        proj = [t.proj],
        other = [t.other],
        total = [t.total]
    )
    # Setting the row name
    temp.rownames = ["First solution"]
    

    # # Storing the number of iterations
    temp.iter = iter
    temp.itrej = itrej
    temp.ns = ns
    temp.pcgit = pcgit
    temp.lpit = lpit
    # ---------- #


    return xstar,
    xstarDsgn,
    u,
    opstop,
    iter,
    itrej,
    F,
    vol,
    ns,
    pcgit,
    lpit,
    t,
    xhist,
    strDens
end

end
