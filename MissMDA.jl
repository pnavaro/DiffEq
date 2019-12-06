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
# # Julia and DifferentialEquations.jl
#

# +
using Plots

x = LinRange(0, 2π, 10)
y = sin.(x) + cos.(x).^2

plot(x, y)

# +
using PyCall
@pyimport scipy.interpolate as interpolate

f = interpolate.interp1d(x, y, kind="cubic")
new_x = LinRange(first(x), last(x), 1000)
plot( new_x, f(new_x))

# +
using RCall

R"""
library(missMDA)
data(orange)
nb = estim_ncpPCA(orange,ncp.max=5)
resMI = MIPCA(orange,ncp=2)
plot(resMI)
"""

