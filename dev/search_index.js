var documenterSearchIndex = {"docs":
[{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"EditURL = \"https://github.com/francescoalemanno/KissABC.jl/blob/master/docs/literate/example_1.jl\"","category":"page"},{"location":"example_1/#A-gaussian-mixture-model-1","page":"Example: Gaussian Mixture","title":"A gaussian mixture model","text":"","category":"section"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"First of all we define our model,","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"using KissABC\nusing Distributions\n\nfunction model(P,N)\n    μ_1, μ_2, σ_1, σ_2, prob=P\n    d1=randn(N).*σ_1 .+ μ_1\n    d2=randn(N).*σ_2 .+ μ_2\n    ps=rand(N).<prob\n    R=zeros(N)\n    R[ps].=d1[ps]\n    R[.!ps].=d2[.!ps]\n    R\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"Let's use the model to generate some data, this data will constitute our dataset","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"parameters = (1.0, 0.0, 0.2, 2.0, 0.4)\ndata=model(parameters,5000)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's look at the data","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"using Plots\nhistogram(data)\nsavefig(\"ex1_hist1.svg\"); nothing # hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"(Image: ex1_hist1)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"we can now try to infer all parameters using KissABC, first of all we need to define a reasonable prior for our model","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"prior=Factored(\n            Uniform(0,2), # there is surely a peak between 0 and 2\n            Uniform(-1,1), #there is a smeared distribution centered around 0\n            Uniform(0,1), # the peak has surely a width below 1\n            Uniform(0,4), # the smeared distribution surely has a width less than 4\n            Beta(2,2) # the number of total events from both distributions look about the same, so we will favor 0.5 just a bit\n        );\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's look at a sample from the prior, to see that it works","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"rand(prior)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"now we need a distance function to compare datasets, this is not the best distance we could use, but it will work out anyway","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"function D(x,y)\n    r=(0.1,0.2,0.5,0.8,0.9)\n    mean(abs,quantile(x,r).-quantile(y,r))\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"we can now run ABCDE to get the posterior distribution of our parameters given the dataset data","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"plan=ABCplan(prior,model,data,D,params=5000)\nres,Δ,converged=ABCDE(plan,0.02,parallel=true,generations=500,verbose=false);\nnothing #hide","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"Has it converged to the target tolerance?","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"print(\"Converged = \",converged)","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"let's see the median and 95% confidence interval for the inferred parameters and let's compare them with the true values","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"function getstats(V)\n    (\n        median=median(V),\n        lowerbound=quantile(V,0.025),\n        upperbound=quantile(V,0.975)\n    )\nend\n\nlabels=(:μ_1, :μ_2, :σ_1, :σ_2, :prob)\nP=[getindex.(res,i) for i in 1:5]\nstats=getstats.(P)\n\nfor is in eachindex(stats)\n    println(labels[is], \" ≡ \" ,parameters[is], \" → \", stats[is])\nend","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"The inferred parameters are close to nominal values","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"","category":"page"},{"location":"example_1/#","page":"Example: Gaussian Mixture","title":"Example: Gaussian Mixture","text":"This page was generated using Literate.jl.","category":"page"},{"location":"reference/#","page":"Reference","title":"Reference","text":"CurrentModule = KissABC","category":"page"},{"location":"reference/#Reference-1","page":"Reference","title":"Reference","text":"","category":"section"},{"location":"reference/#","page":"Reference","title":"Reference","text":"","category":"page"},{"location":"reference/#","page":"Reference","title":"Reference","text":"Modules = [KissABC]","category":"page"},{"location":"reference/#KissABC.KissABC","page":"Reference","title":"KissABC.KissABC","text":"KissABC\n\nModule to perform approximate bayesian computation,\n\nSimple Example: inferring the mean of a Normal distribution\n\nusing KissABC\nusing Distributions\n\nprior=Normal(0,1)\ndata=randn(1000) .+ 1\nsim(μ,other)=randn(1000) .+ μ\ndist(x,y) = abs(mean(x) - mean(y))\n\nplan=ABCplan(prior, sim, data, dist)\nμ_post,Δ = ABCDE(plan, 1e-2)\n@show mean(μ_post) ≈ 1.0\n\nfor more complicated code examples look at https://github.com/francescoalemanno/KissABC.jl/\n\n\n\n\n\n","category":"module"},{"location":"reference/#KissABC.ABCplan","page":"Reference","title":"KissABC.ABCplan","text":"ABCplan(prior, simulation, data, distance; params=())\n\nBuilds a type ABCplan which holds\n\nArguments:\n\nprior: a Distribution to use for sampling candidate parameters\nsimulation: simulation function sim(prior_sample, constants) -> data that accepts a prior sample and the params constant and returns a simulated dataset\ndata: target dataset which must be compared with simulated datasets\ndistance: distance function dist(x,y) that return the distance (a scalar value) between x and y\nparams: an optional set of constants to be passed as second argument to the simulation function\n\n\n\n\n\n","category":"type"},{"location":"reference/#KissABC.Factored","page":"Reference","title":"KissABC.Factored","text":"Factored{N} <: Distribution{Multivariate, MixedSupport}\n\na Distribution type that can be used to combine multiple UnivariateDistribution's and sample from them.\n\nExample: it can be used as prior = Factored(Normal(0,1), Uniform(-1,1))\n\n\n\n\n\n","category":"type"},{"location":"reference/#KissABC.ABC-Tuple{ABCplan,Any}","page":"Reference","title":"KissABC.ABC","text":"ABC(plan, α_target; nparticles = 100, parallel = false)\n\nClassical ABC rejection algorithm.\n\nArguments:\n\nplan: a plan built using the function ABCplan.\nα_target: target acceptance rate for ABC rejection algorithm, nparticles/α will be sampled and only the best nparticles will be retained.\nnparticles:  number of samples from the approximate posterior that will be returned\nparallel: when set to true multithreaded parallelism is enabled\n\n\n\n\n\n","category":"method"},{"location":"reference/#KissABC.ABCDE-Tuple{ABCplan,Any}","page":"Reference","title":"KissABC.ABCDE","text":"ABCDE(plan, ϵ_target; nparticles=100, generations=500, α=0, parallel=false, earlystop=false, verbose=true)\n\nA sequential monte carlo algorithm inspired by differential evolution, very efficient (simpler version of B.M.Turner 2012, https://doi.org/10.1016/j.jmp.2012.06.004)\n\nArguments:\n\nplan: a plan built using the function ABCplan.\nϵ_target: maximum acceptable distance between simulated datasets and the target dataset\nnparticles: number of samples from the approximate posterior that will be returned\ngenerations: total number of simulations per particle\nα: controls the ϵ for each simulation round as ϵ = m+α*(M-m) where m,M = extrema(distances)\nparallel: when set to true multithreaded parallelism is enabled\nearlystop: when set to true a particle is no longer updated as soon as it has reached ϵ_target, this provides a huge speedup, but it can lead to erroneous posterior distribution\nverbose: when set to true verbosity is enabled\n\n\n\n\n\n","category":"method"},{"location":"reference/#KissABC.ABCSMCPR","page":"Reference","title":"KissABC.ABCSMCPR","text":"ABCSMCPR(plan, ϵ_target; nparticles = 100, maxsimpp = 1000.0, α = 0.3, c = 0.01, parallel = false, verbose = true)\n\nSequential Monte Carlo algorithm (Drovandi et al. 2011, https://doi.org/10.1111/j.1541-0420.2010.01410.x).\n\nArguments:\n\nplan: a plan built using the function ABCplan.\nϵ_target: maximum acceptable distance between simulated datasets and the target dataset\nnparticles: number of samples from the approximate posterior that will be returned\nmaxsimpp: average maximum number of simulations per particle\nα: proportion of particles to retain at every iteration of SMC, other particles are resampled\nc: probability that a sample will not be updated during one iteration of SMC\nparallel: when set to true multithreaded parallelism is enabled\nverbose: when set to true verbosity is enabled\n\n\n\n\n\n","category":"function"},{"location":"reference/#KissABC.sample_plan-Tuple{ABCplan,Any,Any}","page":"Reference","title":"KissABC.sample_plan","text":"sample_plan(plan::ABCplan, nparticles, parallel)\n\nfunction to sample the prior distribution of both parameters and distances.\n\nArguments:\n\nplan: a plan built using the function ABCplan.\nnparticles: number of samples to draw.\nparallel: enable or disable threaded parallelism via true or false.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.length-Tuple{Factored}","page":"Reference","title":"Base.length","text":"length(p::Factored) = begin\n\nreturns the number of distributions contained in p.\n\n\n\n\n\n","category":"method"},{"location":"reference/#Base.rand-Tuple{Random.AbstractRNG,Factored}","page":"Reference","title":"Base.rand","text":"rand(rng::AbstractRNG, factoreddist::Factored)\n\nfunction to sample one element from a Factored object\n\n\n\n\n\n","category":"method"},{"location":"reference/#Distributions.pdf-Tuple{Factored,Any}","page":"Reference","title":"Distributions.pdf","text":"pdf(d::Factored, x) = begin\n\nFunction to evaluate the pdf of a Factored distribution object\n\n\n\n\n\n","category":"method"},{"location":"reference/#KissABC.compute_kernel_scales","page":"Reference","title":"KissABC.compute_kernel_scales","text":"compute_kernel_scales(prior::Distribution, V)\n\nFunction for ABCSMCPR whose purpose is to compute the characteristic scale of the perturbation kernel appropriate for prior given the Vector V of parameters\n\n\n\n\n\n","category":"function"},{"location":"reference/#KissABC.deperturb","page":"Reference","title":"KissABC.deperturb","text":"deperturb(prior::Distribution, sample, r1, r2, γ)\n\nFunction for ABCDE whose purpose is computing sample + γ (r1 - r2) + ϵ (the perturbation function of differential evolution) in a way suited to the prior.\n\nArguments:\n\nprior\nsample\nr1\nr2\n\n\n\n\n\n","category":"function"},{"location":"reference/#KissABC.kernel","page":"Reference","title":"KissABC.kernel","text":"kernel(prior::Distribution, c, scale)\n\nFunction for ABCSMCPR whose purpose is returning the appropriate Distribution to use as a perturbation kernel on sample c and characteristic scale\n\nArguments:\n\nprior: prior distribution\nc: sample acting as center of perturbation kernel\nscale: characteristic scale of perturbation kernel\n\n\n\n\n\n","category":"function"},{"location":"reference/#KissABC.kerneldensity","page":"Reference","title":"KissABC.kerneldensity","text":"kerneldensity(prior::Distribution, scales, s1, s2)\n\nFunction for ABCSMCPR whose purpose is returning the probability density of observing s2 under the kernel centered on s1 with scales given by scales and appropriate for prior.\n\n\n\n\n\n","category":"function"},{"location":"reference/#KissABC.perturb","page":"Reference","title":"KissABC.perturb","text":"perturb(prior::Distribution, scales, sample)\n\nFunction for ABCSMCPR whose purpose is perturbing sample according to the appropriate kernel for prior with characteristic scales.\n\n\n\n\n\n","category":"function"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"EditURL = \"https://github.com/francescoalemanno/KissABC.jl/blob/master/docs/literate/index.jl\"","category":"page"},{"location":"#KissABC-1","page":"Basic Usage","title":"KissABC","text":"","category":"section"},{"location":"#Usage-guide-1","page":"Basic Usage","title":"Usage guide","text":"","category":"section"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"The ingredients you need to use Approximate Bayesian Computation:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"A simulation which depends on some parameters, able to generate datasets similar to your target dataset if parameters are tuned\nA prior distribution over such parameters\nA distance function to compare generated dataset to the true dataset","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"We will start with a simple example, we have a dataset generated according to an Normal distribution whose parameters are unknown","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"tdata=randn(1000) .* 0.04 .+ 2;\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we are ofcourse able to simulate normal random numbers, so this constitutes our simulation","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"sim((μ,σ), param) = randn(100) .* σ .+ μ;\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"The second ingredient is a prior over the parameters μ and σ","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"using Distributions\nusing KissABC\nprior=Factored(\n               Uniform(1,3),\n               Truncated(Normal(0,0.1), 0, 100)\n              );\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we have chosen a uniform distribution over the interval [1,3] for μ and a normal distribution truncated over ℝ⁺ for σ.","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"Now all that we need is a distance function to compare the true dataset to the simulated dataset, for this purpose a Kolmogorov-Smirnoff distance is good","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"using StatsBase\nfunction ksdist(x,y)\n    p1=ecdf(x)\n    p2=ecdf(y)\n    r=[x;y]\n    maximum(abs.(p1.(r)-p2.(r)))\nend","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"Now we are all set, we can use ABCDE which is sequential Monte Carlo algorithm with an adaptive proposal, to simulate the posterior distribution for this model, inferring μ and σ","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"plan=ABCplan(prior, sim, tdata, ksdist)\nres,Δ = ABCDE(plan, 0.1, nparticles=2000,generations=150,parallel=true, verbose=false);\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"the parameters we chose are: a tolerance on distances equal to 0.1, a number of simulated particles equal to 200, we enabled Threaded parallelism, and ofcourse the first four parameters are the ingredients we set in the previous steps, the simulated posterior results are in res, while in Δ we can find the distances calculated for each sample. We can now extract the inference results:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"prsample=[rand(prior) for i in 1:2000] #some samples from the prior for comparison\nμ_pr=getindex.(prsample,1) # μ samples from the prior\nσ_pr=getindex.(prsample,2) # σ samples from the prior\nμ_p=getindex.(res,1) # μ samples from the posterior\nσ_p=getindex.(res,2); # σ samples from the posterior\nnothing #hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"and plotting prior and posterior side by side we get:","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"using Plots\na  = stephist(μ_pr,xlims=(1,3),xlabel=\"μ prior\",leg=false,lw=2,normalize=true)\nb  = stephist(σ_pr,xlims=(0,0.3),xlabel=\"σ prior\",leg=false,lw=2,normalize=true)\nap = stephist(μ_p, xlims=(1,3),xlabel=\"μ posterior\",leg=false,lw=2,normalize=true)\nbp = stephist(σ_p, xlims=(0,0.3),xlabel=\"σ posterior\",leg=false,lw=2,normalize=true)\nplot(a,ap,b,bp)\nsavefig(\"inference.svg\"); nothing # hide","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"(Image: inference_plot)","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"we can see that the algorithm has correctly inferred both parameters, this exact recipe will work for much more complicated models and simulations, with some tuning.","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"","category":"page"},{"location":"#","page":"Basic Usage","title":"Basic Usage","text":"This page was generated using Literate.jl.","category":"page"}]
}
