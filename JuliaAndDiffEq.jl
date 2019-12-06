# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.3.0
#   kernelspec:
#     display_name: Julia 1.3.0
#     language: julia
#     name: julia-1.3
# ---

# + [markdown] slideshow={"slide_type": "slide"}
# # DifferentialEquations.jl
#
# website : https://juliadiffeq.org
#
# *Rackauckas, C. & Nie, Q., (2017). DifferentialEquations.jl – A Performant and Feature-Rich Ecosystem for Solving Differential Equations in Julia. Journal of Open Research Software.* 
#
# DifferentialEquations.jl is the core differential equation solver package in Julia. In a nutshell, it provides solvers for:
#
# - ODEs
# - DAEs
# - SODEs, SDAEs
# - Discrete stochastic (Gillespie) equations, mixed with ODEs/SDEs (jump diffusions)
# - DDEs
# - PDEs
#
# It provides wrappers to the common C++ and Fortran libraries, along with native Julia implementations with efficient generic algorithms.
#
# 40 contributors
# -

using DifferentialEquations, Plots

# + [markdown] slideshow={"slide_type": "slide"}
# ## Case Study: Lotka-Volterra in many ways
#
# First, the basic ODE with functions

# + slideshow={"slide_type": "slide"}
function f(du,u,p,t)
  x, y = u
  α, β, γ, δ = p
  du[1] = α*x - β*x*y
  du[2] = -γ*y + δ*x*y
end

p = (1.5,1.0,3.0,1.0)
u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob1 = ODEProblem(f,u0,tspan,p)
sol = solve(prob1)
plot(sol)
# -

plot(sol; vars=(1, 2))

# + [markdown] slideshow={"slide_type": "slide"}
# ## DSLs and Metaprogramming for Syntactic Sugar
#
# Now let's use a DSL defined by a macro:
# -

using ParameterizedFunctions

# +
lorenz63 = @ode_def Lorenz begin
  dx = σ*(y-x)
  dy = ρ*x-y-x*z
  dz = x*y-β*z
end σ β ρ 

u0 = [1., 5., 10.]
tspan = (0., 100.)
σ, β, ρ = 10.0, 8.0/3.0, 28.0
p = [σ, β, ρ]
prob = ODEProblem(lorenz63, u0, tspan, p)
sol = solve(prob);
# -

plot(sol, plotdensity=10000, vars=(1, 2, 3))

# + [markdown] slideshow={"slide_type": "slide"}
# ## Optimize the Solver By Choosing from the Largest Set of Methods
#
# http://docs.juliadiffeq.org/latest/solvers/ode_solve.html
#
# https://github.com/JuliaDiffEq/DiffEqBenchmarks.jl
# -

using BenchmarkTools

# + slideshow={"slide_type": "slide"}
@benchmark solve(prob,Vern9())

# + slideshow={"slide_type": "fragment"}
using LSODA
@benchmark solve(prob1,lsoda()) # Default method of SciPy, deSolve, etc.

# + [markdown] slideshow={"slide_type": "slide"}
# ## Generic Algorithms to Propogate Uncertainty

# + slideshow={"slide_type": "fragment"}
using Measurements
u0 = [1.0 ± 0.0 ,1.0 ± 0.0]
p = (1.5 ± 0.0,1.0 ± 0.1,3.0 ± 0.2,1.0 ± 0.1)
tspan = (0.0,10.0)
prob_error = ODEProblem(f,u0,tspan,p)
sol = solve(prob_error,Tsit5(), saveat=0.2)
plot(sol)

# + [markdown] slideshow={"slide_type": "slide"}
# ## Generic Algorithms to Calculate Sensitivities

# + slideshow={"slide_type": "fragment"}
using ForwardDiff: Dual
p1dual = Dual{Float64}(1.5, (1.0, 0.0))
p2dual = Dual{Float64}(1.0, (0.0, 1.0))
pdual = (p1dual, p2dual, 3.0, 1.0)
u0 = [Dual{Float64}(1.0, (0.0, 0.0)),Dual{Float64}(1.0, (0.0, 0.0))]
prob_dual = ODEProblem(f,u0,tspan,pdual)
sol_dual = solve(prob_dual,Tsit5(), saveat=0.2)

timepoints = [i for i in sol_dual.t]
sensitivity_forward_diff = [i[1].partials.values[1] for i in sol_dual.u]
Plots.plot(timepoints,sensitivity_forward_diff,title="dx/da",lw=3,xaxis="Time")

# + [markdown] slideshow={"slide_type": "slide"}
# ## Generic Algorithms for Arbitrary Precision

# + slideshow={"slide_type": "fragment"}
p = big.((1.5,1.0,3.0,1.0))
u0 = big.([1.0,1.0])
tspan = big.((0.0,10.0))
prob1 = ODEProblem(f,u0,tspan,p)
sol = solve(prob1,Vern9(),abstol=1e-25,reltol=1e-25)
sol[10]

# + [markdown] slideshow={"slide_type": "slide"}
# ## Generic Algorithms for Units

# + slideshow={"slide_type": "slide"}
p = (1.5,1.0/u"kg",3.0,1.0/u"kg")./u"s"
u0 = [1.0u"kg",1.0u"kg"]
tspan = (0.0u"s",10.0u"s")
prob_units = ODEProblem(f,u0,tspan,p)
sol = solve(prob_units,Tsit5())
sol.t[10],sol[10]

# + [markdown] slideshow={"slide_type": "slide"}
# # Other Features of DifferentialEquations.jl

# + [markdown] slideshow={"slide_type": "slide"}
# ## The solution is a continuous function by default

# + slideshow={"slide_type": "fragment"}
sol(5.5u"s")

# + [markdown] slideshow={"slide_type": "slide"}
# ## Easily Change to Stochastic Differential Equations

# + slideshow={"slide_type": "fragment"}
using DifferentialEquations
function g(du,u,p,t)
  du[1] = 0.2u[1]
  du[2] = 0.2u[2]
end
p = (1.5,1.0,3.0,1.0); u0 = [1.0;1.0]
tspan = (0.0,10.0)
prob1 = SDEProblem(f,g,u0,tspan,p)
sol = solve(prob1)
using Plots; plot(sol)

# + [markdown] slideshow={"slide_type": "slide"}
# ## And Delay Differential Equations

# + slideshow={"slide_type": "fragment"}
function f(du,u,h,p,t)
  x, y = u
  α,β,γ,δ = p
  du[1] = α*h(p,t-1)[1] - β*x*y
  du[2] = -γ*y + δ*x*y
end
p = (1.5,1.0,3.0,1.0); u0 = [1.0;1.0]
tspan = (0.0,10.0)
_h(p,t) = ones(2)
prob1 = DDEProblem(f,u0,_h,tspan,p)
sol = solve(prob1)
using Plots; plot(sol)
