# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,jl:light
#     text_representation:
#       extension: .jl
#       format_name: light
#       format_version: '1.4'
#       jupytext_version: 1.2.1
#   kernelspec:
#     display_name: Julia 1.1.1
#     language: julia
#     name: julia-1.1
# ---

# +
using ModelingToolkit, DifferentialEquations, Plots

# Define some variables
@parameters t σ ρ β
@variables x(t) y(t) z(t)
@derivatives D'~t
# -

eqs = [D(x) ~ σ*(y-x),
       D(y) ~ x*(ρ-z)-y,
       D(z) ~ x*y - β*z]

de = ODESystem(eqs)


generate_function(de, [x,y,z], [σ,ρ,β])

f = ODEFunction(de, [x,y,z], [σ,ρ,β])

u0 = [1.0,0.0,0.0]
tspan = (0.0,100.0)
p = [10.0,28.0,8/3]
prob = ODEProblem(f,u0,tspan,p)

sol = solve(prob)

plot(sol,vars=(1,2,3))





