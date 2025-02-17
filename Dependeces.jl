using Pkg

# Ativa o ambiente do projeto
Pkg.activate(".")

# Lista de pacotes e suas versões
pacotes = Dict(
    "AMD" => v"0.5.3",
    "DataFrames" => v"1.7.0",
    "HiGHS" => v"1.10.2",
    "IncompleteLU" => v"0.2.1",
    "IterativeSolvers" => v"0.9.4",
    "JuMP" => v"1.23.3",
    "MAT" => v"0.10.7",
    "Plots" => v"1.40.8"
)

# Instala cada pacote na versão especificada
for (pacote, versao) in pacotes
    Pkg.add(PackageSpec(name=pacote, version=versao))
end
