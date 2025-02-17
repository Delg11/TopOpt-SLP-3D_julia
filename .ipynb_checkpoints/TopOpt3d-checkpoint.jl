using MAT
using Pkg

# Incluindo o arquivo do módulo
include("StartStrSLP.jl")
include("TotalStr.jl")
include("Display3D.jl")

using .Display3DModule  # O ponto indica um módulo local
using .StartStrSLPModule  # O ponto indica um módulo local
using .TotalStrModule  # O ponto indica um módulo local

println("Starting...")
# Definição da estrutura com tipos corretos
mutable struct Material
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
    supp::Union{Matrix{Float64},Nothing}
    freedofs::Union{Matrix{Int64},Nothing}
    loads::Union{Matrix{Float64},Nothing}
    f::Union{Matrix{Float64},Nothing}
    fix::Union{Matrix{Int},Nothing}
    freeDens::Union{Matrix{Int},Nothing}
    fixedDens::Union{Matrix{Int},Nothing}
    fixedDensVal::Union{Matrix{Float64},Nothing}
    freeDensG::Union{Matrix{Int},Nothing}

end
mutable struct MaterialDens
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
    fde::Union{Matrix{Float64},Nothing}
    freeDens::Union{Matrix{Float64},Nothing}
    fixedDens::Union{Matrix{Float64},Nothing}
    fixedDensVal::Union{Matrix{Float64},Nothing}
end
# Estruturas para armazenar os parâmetros

# Estrutura para elementos finitos
mutable struct Element
    deg::Int           # Element degree (polynomial degree of the shape functions)
    type::Int         # Element type (1 = Lagrange, 2 = Serendipity)
    increasedeg::Bool  # If we will solve the problem increasing the degree
    maxdeg::Int        # Maximum element degree 
    fix::Int           # Defines the way we choose variables to be fixed when increasing the element degree
    fixit::Int         # Number of outer iterations to choose fixed variables
    fixtl::Float64     # Tolerance for approximately void elements
    fixtu::Float64     # Tolerance for approximately solid elements
    fixnb::Bool        # Choose all void elements or just those surrounded by void elements
    fixdofs::Int       # Fix displacements of void elements
    fixdsgn::Bool      # Choose void elements based on near-zero design variables or densities
    ltdsgn::Float64    # Maximum value a neighbor may have when fixing design variables to zero
    utdsgn::Float64    # Minimum value when fixing design variables to one
    tolgdsgn::Float64   # Tolerance for the gradient when fixing design variables
end

# Estrutura para parâmetros SLP (Sequential Linear Programming)
struct SLPParameters
    tolS::Float64      # Threshold for the step norm in SLP
    tolG::Float64      # Threshold for the projected gradient norm in SLP
    tolF::Float64      # Threshold for the reduction of the objective function
    maxcount::Int      # Stopping criterion based on step norm
    maxiter::Int       # Maximum number of iterations for the SLP
    maxit12::Int       # Maximum number of iterations for p = 1, 2
    lpsolver::Int      # Linear programming solver type
    delta0::Float64    # Initial trust region radius
    eta::Float64       # Acceptance parameter
    rho::Float64       # Parameter to increase trust region radius
    alphaR::Float64    # Factor to decrease trust region radius on rejection
    alphaA::Float64    # Factor to increase trust region radius on acceptance
end

# Estrutura para a estratégia de projeção de densidade
mutable struct ProjectionStrategy
    op::Bool           # Define if densities are rounded to 0 or 1
    nearlim::Bool      # Round to nearest limit (0 or 1)
    maxang::Float64    # Largest angle allowed for rounding
    maxit::Int         # Maximum number of SLP calls when op = true
    tolV::Float64      # Maximum violation of the volume fraction
    tolN::Float64      # Maximum fraction of density changes allowed
    opfilter::Int      # Filter used after projection
    rmin::Float64      # Filter radius used in projection
    gthresh::Float64    # Gradient threshold for considering near zero
    ut::Float64        # Upper limit threshold
    lt::Float64        # Lower limit threshold
    vumin::Float64     # Smallest density that can be rounded to 1
    vlmax::Float64     # Largest density that can be rounded to 0
    heavi::Bool     # If smooth Heaviside projection is applied
    eta::Float64       # Initial value for the Heaviside filter function
    beta::Float64      # Initial beta parameter value
    dbeta::Float64     # Increasing factor for the beta parameter
    betamax::Float64   # Maximum beta parameter value
    deg::Bool          # Use linear elements on SLP calls after rounding densities
end

# Estrutura para considerar problemas com simetrias
struct Symmetry
    xy::Bool           # If there is symmetry with respect to the xy plane
    yz::Bool           # If there is symmetry with respect to the yz plane
    xz::Bool           # If there is symmetry with respect to the xz plane
end

# Estrutura para parâmetros do PCG (Preconditioned Conjugate Gradient)
struct PCGParameters
    opu0::Bool         # Use previous vector u as initial guess for PCG
    tolPCG::Float64    # Convergence tolerance for PCG
    maxiterPCG::Int    # Maximum number of iterations for PCG
end

# Estrutura para parâmetros multigrid
struct MultiGrid
    ngrids::Int         # Number of grids
    cycle::Int         # Cycle type (0 - Vcycle, 1 - Wcycle, 2 - FullVCycle)
    smoother::Int      # Smoother type (0 - Jacobi, 1 - Gauss-Seidel/SOR, 2 - SSOR)
    omega::Float64     # Smoother relaxation parameter
    smoothiter1::Int   # Number of pre-smoothing iterations
    smoothiter2::Int   # Number of post-smoothing iterations
    tolMG::Float64     # Tolerance for CG with multigrid
    maxiterMG::Int     # Maximum iterations for CG with multigrid
    theta::Float64     # Strong connections parameter
    optheta::Int       # Choose if 'theta' is a tolerance or a count of connections
    nt::Int            # Number of test space vectors
    tolr::Float64      # Tolerance for test space generation
    tolq::Float64      # Tolerance for test space generation
    itp::Int           # Maximum iterations on prolongation DPLS
    kappa::Float64     # Condition number tolerance for prolongation DPLS
    opcond::Int        # Preconditioner option for test space generation and PCG
    nmaxchol::Int      # Maximum dimension for Cholesky factorization on coarsest grid
    opchol::Int        # Action when dimension exceeds 'nmaxchol'
    Anbatch::Int       # Batch used in An computation when genK = false
end

# Estrutura para parâmetros de multiresolução
struct MultiResolution
    op::Bool           # If the multiresolution method will be used
    n::Int             # Number of density elements per finite element dimension
    d::Int             # Number of design variable elements per finite element dimension
    x0::Bool           # Use coarse problem solution as initial densities
    interp::Bool       # If displacements will be interpolated for gradient calculation
end

# Contendo todas as configurações em uma estrutura principal
#struct Settings
#    elem::Element               # Element parameters
#    slp::SLPParameters           # SLP parameters
#    prj::ProjectionStrategy      # Density projection strategy parameters
#    sym::Symmetry               # Symmetry parameters
#    pcgp::PCGParameters          # PCG parameters
#    mg::MultiGrid               # Multigrid parameters
#    mr::MultiResolution          # Multiresolution parameters
#    opsolver::Int               # Linear system solver option
#    genK::Bool                  # Generate global stiffness matrix?
#end

# Estrutura para calcular resultados extras com base na solução obtida
struct ExtraResults
    correctF::Bool  # Calculate the "correct" function value when using multiresolution or degree > 1
    postopt::Bool    # Solve the problem again on the fine mesh with multiresolution solution as initial guess
    roundsol::Bool   # Round greatest densities to 1 and others to 0, calculate the objective function
    solidsol::Bool   # Calculate the objective function of the full solid structure
end
function load(file_name::String)
    # Ler o arquivo .mat e armazenar todas as variáveis
    vars = matread(file_name)

    # Verifica se a variável 'str' existe
    if haskey(vars, "str")
        str = vars["str"]

        # Inicializa um dicionário para os dados da estrutura
        data = Dict{Symbol,Union{Float64,Matrix{Float64},Nothing}}()  # Adicione Matrix{Float64} a Union

        # Obtém os nomes dos campos da estrutura
        field_names = fieldnames(Material)
        # Converte os nomes dos campos de símbolos para strings
        field_names_str = [string(field) for field in field_names]

        # Preenche o dicionário com as variáveis que existem
        for (key, value) in str
            # Verifica se a chave existe nos nomes dos campos da estrutura (como string)
            if key in field_names_str
                # Converte a chave para um símbolo
                symbol_key = Symbol(key)
                # Atribui o valor ao dicionário
                data[symbol_key] = value
            else
                println(
                    "A variável $key não é um campo da estrutura Material e será ignorada.",
                )
            end
        end

        # Cria uma nova instância de Material com os dados coletados
        material = Material(
            get(data, :l, nothing),
            get(data, :h, nothing),
            get(data, :w, nothing),
            get(data, :nelx, nothing),
            get(data, :nely, nothing),
            get(data, :nelz, nothing),
            get(data, :E, nothing),
            get(data, :nu, nothing),
            get(data, :el, nothing),
            get(data, :eh, nothing),
            get(data, :ew, nothing),
            get(data, :nx, nothing),
            get(data, :ny, nothing),
            get(data, :nz, nothing),
            get(data, :nelem, nothing),
            get(data, :nnodes, nothing),
            get(data, :supp, nothing),
            get(data, :freedofs, nothing),
            get(data, :loads, nothing),
            get(data, :f, nothing),
            get(data, :fix, nothing),
            get(data, :freeDens, nothing),
            get(data, :fixedDens, nothing),
            get(data, :fixedDensVal, nothing),
            get(data, :freeDensG, nothing),
        )

        return material  # Retorna a estrutura preenchida
    else
        println("A variável 'str' não foi encontrada no arquivo.")
        return nothing  # Retorna nothing se 'str' não estiver presente
    end
end
# Main program to solve the three-dimensional structural topology optimization problem of minimum compliance, using a sequential linear programming algorithm #

# Problem data (str)
# str=load("Tests/cb30x10x10half.mat"); # cantilever beam
# str=load("Tests/cb48x16x16.mat");
# str=load("Tests/cb48x16x16half.mat");
# str=load("Tests/cb60x20x20.mat"); 
# str=load("Tests/cb72x24x24.mat"); 
# str=load("Tests/cb90x30x30.mat"); 
# str=load("Tests/cb96x32x32.mat"); 
# str=load("Tests/cb96x32x32half.mat");
# str=load("Tests/cb120x40x40.mat");
# str=load("Tests/cb120x40x40half.mat");
# str=load("Tests/cb144x48x48.mat");
# str=load("Tests/cb144x48x48half.mat");
# str=load("Tests/cb150x50x50.mat");
# str=load("Tests/cb168x56x56.mat");
# str=load("Tests/cb180x60x60.mat");
# str=load("Tests/cb192x64x64.mat");
# str=load("Tests/cb192x64x64half.mat");
# str=load("Tests/cb60x30x2.mat");
str = load("Tests/mbb96x16x16quarter.mat"); # mbb beam
# str=load("Tests/mbb120x20x20quarter.mat");
# str=load("Tests/mbb192x32x32quarter.mat");
# str=load("Tests/mbb240x40x40quarter.mat");
# str=load("Tests/mbb288x48x48quarter.mat");
# str=load("Tests/mbb384x64x64quarter.mat");
# str=load("Tests/mbb96x16x1half.mat");
# str=load("Tests/tt16x80x16quarter.mat"); # transmission tower
# str=load("Tests/tt32x160x32quarter.mat");
# str=load("Tests/tt48x240x48quarter.mat");
# str=load("Tests/tt64x320x64quarter.mat");
# str=load("Tests/eb60x20x30.mat"); # engine bracket
# str=load("Tests/eb48x16x24half.mat"); 
# str=load("Tests/hc24x24x24quarter.mat"); # cube with concentrated str=load at the bottom 
# str=load("Tests/hc32x16x32quarter.mat");
# str=load("Tests/hc96x48x96quarter.mat");
# str=load("Tests/bt16x64x16.mat"); # building with torsion
# str=load("Tests/bt32x128x32.mat");  
# str=load("Tests/ls48x48x16half.mat"); # L-shaped beam
# str=load("Tests/ls144x144x48half.mat");
# str=load("Tests/ls192x192x64half.mat");
# str=load("Tests/bd192x48x24quarter.mat"); str.f = str.f/norm(str.f); # bridge

# Maximum volume fraction of the domain that the structure can occupy
volfrac = 0.2; # test
# volfrac = 0.2; # cantilever beam 
# volfrac = 0.2; # mbb beam
# volfrac = 0.15; # transmission tower 
# volfrac = 0.15; # engine bracket 
# volfrac = 0.16; # 1/2 cube with concentrated str=load at the bottom 
# volfrac = 0.1; # building with torsion
# volfrac = 0.15; # bridge (includes the fixed elements)
# volfrac = 0.1; # L-shaped beam (includes the fixed elements)

# Filter radius
rmin = 1.5; # test
# rmin = 1.4; # cb30x10x10 # cantilever beam
# rmin = 1.5; # cb48x16x16
# rmin = 1.5; # cb60x20x20
# rmin = 1.5; # cb72x24x24
# rmin = 2.0; # cb90x30x30
# rmin = 2.2; # cb96x32x32 
# rmin = 2.5; # cb120x40x40 
# rmin = 3.5; # cb144x48x48 
# rmin = 4.0; # cb150x50x50
# rmin = 4.2; # cb168x56x56
# rmin = 4.7; # cb180x60x60 
# rmin = 5.2; # cb192x64x64
# rmin = 2.0; # mbb96x16x16 # mbb beam
# rmin = 2.5; # mbb120x20x20
# rmin = 5.0; # mbb192x32x32
# rmin = 5.5; # mbb240x40x40 
# rmin = 6.0; # mbb288x48x48
# rmin = 8.0; # mbb384x64x64
# rmin = 1.1; # tt16x80x16 # transmission tower
# rmin = 2.0; # tt32x160x32
# rmin = 2.2; # tt40x200x40
# rmin = 1.5; # eb60x20x30 # engine bracket
# rmin = 1.5; # eb48x16x24
# rmin = 1.5; # hc32x16x32 # half cube with concentrated str=load at the bottom
# rmin = 3.0; # hc96x48x96
# rmin = 1.5; # bt16x64x16 # building with torsion
# rmin = 3.0; # bt32x128x32 
# rmin = 1.5; # ls48x48x16 # L-shaped beam
# rmin = 3.0; # ls144x144x48
# rmin = 1.5; # bd192x48x24 # bridge

# Definição dos parâmetros

p = 3; # Penalty parameter for the SIMP model
p123 = false; # Defines if we will set p = 1, 2 and 3 (p123 = true) or use only one value of p (p123 = false)
femin = 1.0e-6; # Stiffness factor of the void material (E_(void) = femin*E)
emin = femin * str.E; # Young's modulus of the void material
opfilter = 1; # Filter option (0 = no filter, 1 = weighted average density filter, 2 = average density filter) 
volineq = true; # Defines the type of the volume constraint (true = "less than or equal to", false = "equal to")

# Finite element characteristics and parameters 
elemdeg = 1; # Element degree (polynomial degree of the shape functions). It is possible to use degree 1, 2 or 3
elemtype = 1; # Element type (1 = Lagrange, 2 = Serendipity)
elemincreasedeg = false; # Defines if we will solve the problem increasing the degree of the elements until reach 'elemmaxdeg'
elemmaxdeg = 2; # Maximum element degree 
elemfix = 4; # Defines the way we choose variables to be fixed when increasing the element degree
#(0 = don't choose variables to be fixed, 1 = choose only at the first iteration, 2 = choose in all iterations, 3 = choose every 'elemfixit' iterations, 4 = choose in iterations 1 and 'elemfixit')
elemfixit = 5; # Number of outer iterations to be accomplished until choose again the variables that will be fixed
elemfixtl = 1e-6; # Tolerance to find approximately void elements (the ones full of densities <= elemfixtl)
elemfixtu = 0.9; # Tolerance to find approximately solid elements (the ones full of densities >= elemfixtu)
elemfixnb = false; # Defines if we will choose all void elements to be fixed (fixnb = true) or just the ones surrounded by void elements (fixnb = false)
elemfixdofs = 1; # Defines if we will fix displacements of void elements eliminating the degrees of freedom from the linear system 
#(0 = don't fix any displacements, 1 = don't fix the displacements of nodes shared by neighbor elements, 2 = fix the displacements of all nodes of void elements
elemfixdsgn = false; # Defines if we will choose void elements looking for the ones full of approximately zero design variables (fixdsgn = true) or densities (fixdsgn = false)
elemltdsgn = 0.3; # Maximum value a neighbor may have when fixing design variables to zero
elemutdsgn = 0.7; # Minimum value a neighbor may have when fixing design variables to one
elemtolgdsgn = 1e-6; # Tolerance for the gradient when fixing design variables

elem = Element(
    elemdeg,
    elemtype,
    elemincreasedeg,
    elemmaxdeg,
    elemfix,
    elemfixit,
    elemfixtl,
    elemfixtu,
    elemfixnb,
    elemfixdofs,
    elemfixdsgn,
    elemltdsgn,
    elemutdsgn,
    elemtolgdsgn,
)

# SLP parameters
slptolS = 1e-4; # Threshold for the step norm obtained by the SLP algorithm
slptolG = 1e-3; # Threshold for the projected gradient norm obtained by the SLP algorithm
slptolF = 5e-2; # Threshold for the actual reduction of the objective function obtained by the SLP algorithm
slpmaxcount = 3; # Number of times that the step norm (or projected gradient norm) needs to reach a value less or equal than the threshold before stopping the SLP algorithm (stopping criterion)
slpmaxiter = 500; # Maximum number of iterations (stopping criterion)
slpmaxit12 = 15;  # Maximum number of iterations for p = 1, 2
slplpsolver = 0; # Linear programming solver (0 = dual simplex, 1 = interior point method)
slpdelta0 = 0.1; # Initial trust region radius
slpeta = 0.1; # Parameter used to accept the step (s is accepted if ared >= 0.1*pred)
slprho = 0.5; # Parameter used to increase the trust region radius (delta is increased if ared >= 0.5*pred)
slpalphaR = 0.25; # Factor used to decrease the trust region radius when the step is rejected
slpalphaA = 2.0; # Factor used to increase the trust region radius when the step is accepted

slp = SLPParameters(
    slptolS,
    slptolG,
    slptolF,
    slpmaxcount,
    slpmaxiter,
    slpmaxit12,
    slplpsolver,
    slpdelta0,
    slpeta,
    slprho,
    slpalphaR,
    slpalphaA,
)
# Parameters used for the density projection strategy 
prjop = false; # Defines if the densities are to be rounded to 0 or 1 at the end of the SLP algorithm.
prjnearlim = true; # Try to round the variables to the nearest limit (0 or 1) prior to using the gradient.
prjmaxang = 89.9; # Largest angle (in degrees) allowed between (xnew-x) and -g when rounding the densities.
prjmaxit = 10; # Maximum number of calls to the SLP algorithm when prjop = true.
prjtolV = 0.005; # Maximum violation of the volume fraction allowed after projection.
prjtolN = 0.01; # Maximum fraction of element density changes between consecutive projected solutions. 
prjopfilter = 1; # Filter used after the first projection (0 = no filter, 1 = weighted average density filter, 2 = average density filter).
prjrmin = 1.1; # Filter radius to be used when calling StrSLP after a projection. 
prjgthresh = 1e-4; # Gradient threshold (g(i) is considered to be near 0 if abs(g(i)) <= gthresh*norm(g,inf)).
prjut = 0.95; # Upper limit threshold (densities greater or equal to ut are always rounded to 1)
prjlt = 0.05; # Lower limit threshold (densities lower or equal to lt are always rounded to 0)
prjvumin = 0.7; # Smallest density that can be rounded to 1
prjvlmax = 0.3; # Largest density that can be rounded to 0
prjheavi = true; # Indicates if the smooth Heaviside projection will be applied to round the densities.
prjeta = 0.25; # Initial value of the eta parameter, used to compute the Heaviside filter function.
prjbeta = 1.0; # Initial value of the beta parameter, used to compute the Heaviside filter function.
prjdbeta = 2.0; # Increasing factor for the beta parameter (beta(k+1) = dbeta*beta(k)).
prjbetamax = 1000.0; # Maximum value of the Heaviside beta parameter.
prjdeg = false; # Defines if we will use linear elements on the calls to the SLP algorithm after rounding the densities. 

prj = ProjectionStrategy(
    prjop,
    prjnearlim,
    prjmaxang,
    prjmaxit,
    prjtolV,
    prjtolN,
    prjopfilter,
    prjrmin,
    prjgthresh,
    prjut,
    prjlt,
    prjvumin,
    prjvlmax,
    prjheavi,
    prjeta,
    prjbeta,
    prjdbeta,
    prjbetamax,
    prjdeg,
)
# Parameters for considering the problem with symmetries 
symxy = true; # Defines if the problem domain has a symmetry with respect to the xy plane (true for cb, mbb, tt and hc problems)
symyz = true; # Defines if the problem domain has a symmetry with respect to the yz plane (true for mbb, tt and hc problems)
symxz = false; # Defines if the problem domain has a symmetry with respect to the xz plane
sym = Symmetry(symxy, symyz, symxz)

opsolver = 0; # Linear system solver option
#0 = Cholesky factorization, 
#1 = CG w/ preconditioner being the diagonal of K, 
#2 = CG w/ preconditioner being the incomplete Cholesky factorization of K,
#3 = CG w/ Algebraic Multigrid,
#4 = CG w/ Geometric Multigrid. 

# PCG parameters
pcgpopu0 = true; # Adoption of the previous vector u as an initial guess for pcg
pcgptolPCG = 1e-8; # Convergence tolerance for pcg
pcgpmaxiterPCG = 10000; # Maximum number of iterations of pcg

pcgp = PCGParameters(pcgpopu0, pcgptolPCG, pcgpmaxiterPCG)

genK = true; # Defines if the global stiffness matrix will be explicitly generated
# You can set genK to false whenever opsolver = 1, 3 or 4 and the problem is huge
push!(LOAD_PATH, "./WOK")


# Multigrid parameters 
mgngrids = 3; # Number of grids
mgcycle = 1; # Cycle type (0 - Vcycle, 1 - Wcycle, 2 - FullVCycle)
mgsmoother = 0; # Smoother (0 - Jacobi, 1 - Gauss-Seidel/SOR, 2 - SSOR)
mgomega = 0.5; # Smoother relaxation parameter 
mgsmoothiter1 = 1; # Number of pre-smooth iterations 
mgsmoothiter2 = 1; # Number of post-smooth iterations 
mgtolMG = 1e-8; # Tolerance for CG w/ multigrid
mgmaxiterMG = 5000; # Maximum number of iterations for CG w/ multigrid 
mgtheta = 20; # Strong connections parameter 
mgoptheta = 1; # Choose if 'theta' will be used as (0 - a tolerance, 1 - the maximum number of strong connections per node)
mgnt = 30; # Number of test space vectors
mgtolr = 1e-1; # Tolerance for the test space generation 
mgtolq = 1e-2; # Tolerance for the test space generation 
mgitp = 2; # Maximum number of iterations on prolongation DPLS 
mgkappa = 100; # Tolerance for the condition number on prolongation DPLS
mgopcond = 1; # Preconditioner option for the test space generation and for PCG to solve the system on the coarsest grid (0 - diag, 1 - ichol)
mgnmaxchol = 60000; # Maximum system dimension to be able to use the Cholesky factorization on the coarsest grid
mgopchol = 0; # Action adopted when the dimension of the system on the coarsest grid is greater than 'nmaxchol' (0 - increase ngrids, 1 - use PCG)
mgAnbatch = 4096; # Batch used in the computation of An when genK = false. An is assembled in batches of Anbatch elements.
mg = MultiGrid(
    mgngrids,
    mgcycle,
    mgsmoother,
    mgomega,
    mgsmoothiter1,
    mgsmoothiter2,
    mgtolMG,
    mgmaxiterMG,
    mgtheta,
    mgoptheta,
    mgnt,
    mgtolr,
    mgtolq,
    mgitp,
    mgkappa,
    mgopcond,
    mgnmaxchol,
    mgopchol,
    mgAnbatch,
)

# Multiresolution parameters
mrop = false; # Defines if the multiresolution method will be used.
mrn = 2; # Number of density elements on each direction of each finite element.
mrd = 2; # Number of design variable elements on each direction of each finite element.
mrx0 = false; # Defines if the solution of the coarse problem will be used as initial element densities.
mrinterp = false; # Defines if the displacements will be interpolated to calculate the gradient.
mr = MultiResolution(mrop, mrn, mrd, mrx0, mrinterp)

strDens = MaterialDens(
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
# Apply an SLP algorithm to solve the problem
xstar, xstarDsgn, u, opstop, iter, itrej, F, vol, ns, pcgit, lpit, time, xhist, strDens =
    StartStrSLP(
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

# Calculate extra results based on the solution obtained
exopcorrectF = false; # Calculate the "correct" function value when multiresolution or an element degree greater than 1 is used
exoppostopt = false; # Solve the problem again on the fine mesh, with the multiresolution solution as initial guess
exoproundsol = false; # Round the greatest densities to 1 until filling the volume and the others to 0, then calculate the objective function of the rounded solution
exopsolidsol = false; # Calculate the objective function of the full solid structure (with all densities equal to 1)

exop = ExtraResults(exopcorrectF, exoppostopt, exoproundsol, exopsolidsol)

# Verifique as condições
if exop.correctF || exop.postopt || exop.roundsol || exop.solidsol
    exres = ExtraResults(
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
        xstarDsgn,
        xstar,
        exop,
    )
end

# Exibir uma visão 3D da estrutura ótima
if mr.op
    if sym.xy || sym.yz || sym.xz
        xstartotal = TotalStr(strDens, xstar, sym, prj.op)
    else
        Display3D(strDens, xstar, prj.op)
    end
else
    if sym.xy || sym.yz || sym.xz
        xstartotal = TotalStr(str, xstar, sym, prj.op)
    else
        Display3D(str, xstar, prj.op)
    end
end
