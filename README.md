# TopOpt-SLP-3D_julia
A Julia translation of the TopOpt-SLP-3D Matlab repository for 3D structural topology optimization created by Alfredo Vitorino and Francisco A. M. Gomes. It uses Sequential Linear Programming (SLP) to maximize stiffness under a volume constraint. Advanced features like multigrid and multiresolution are not yet implemented.

# TopOpt-SLP-3D-Julia  

## Overview  
This repository provides a **Julia** implementation of a structural topology optimization algorithm, translated from the original [TopOpt-SLP-3D](https://github.com/AlfredoVitorino/TopOpt-SLP-3D) written in **Matlab**.  

Structural topology optimization is a mathematical methodology used to determine the optimal material distribution within a design domain to maximize stiffness under external loads, while satisfying a volume constraint. It aids in identifying suitable initial designs for structures and accelerating the design process.  

The optimization problem is solved using a **Sequential Linear Programming (SLP)** method with a stopping criterion based on first-order optimality conditions.  

### Current Features  
- Basic implementation of the **SLP optimization framework** for 3D topology optimization problems.  
- Support for solving structural equilibrium equations using standard solvers.  
- Implementation of a volume constraint and material penalization via SIMP (Solid Isotropic Material with Penalization).  

### Limitations  
This first version is incomplete and does not yet include advanced features such as:  
- Multiresolution schemes.  
- Multigrid preconditioners.  
- Adaptive strategies or thresholding techniques.  

## Future Work  
Upcoming updates will progressively add these advanced functionalities to match the original Matlab implementation.  

## License  
This project is licensed under a [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).  

Credits for the original implementation go to **Alfredo Vitorino** and **Francisco A. M. Gomes**.  

