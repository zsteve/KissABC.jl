using KissABC
using AbstractMCMC
using Statistics
using Test
using Random
using LinearAlgebra
using Plots 
Random.seed!(1)

d = Factored(Dirichlet(rand(2)), Dirichlet(rand(2)))
@test pdf(d, ([-1, -1], [3, 3])) == 0
@test logpdf(d, ([-1, -1], [3, 3])) == -Inf
@test length(d) == 2

P = zeros(5, 5) .+ 0.1
P[1:3, 1:3] .= 1
P[3:5, 3:5] .= 1
P = P./sum(P; dims = 2)

prior = Factored([Dirichlet(1.0 .+ P[i, :]) for i = 1:5]...)

function dist(P, Q)
    Q_ = hcat(Q...)'
    d=  norm(P.-Q_, 2)^2
    return d
end

# approx_density = ApproxKernelizedPosterior(prior,x -> dist(P, x), 1e-2)
# res = KissABC.sample(approx_density,AIS(500),1000,ntransitions=100); 
res = smc(prior,x -> dist(P, x), nparticles = 5_000, verbose = true, alpha = 0.5, epstol = 1e-2).P; 

Phat = hcat(pmean.(res)...)'
Phat_std = hcat(pstd.(res)...)'
plot(heatmap(Phat), heatmap(P))


