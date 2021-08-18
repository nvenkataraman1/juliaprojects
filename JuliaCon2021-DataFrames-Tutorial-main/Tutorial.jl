using CSV
using CategoricalArrays
using Chain
using DataFrames
import Downloads
using GLM
using Plots
using Random
using StatsPlots
using Statistics
using Bootstrap

ENV["LINES"] = 20
ENV["COLUMNS"] = 1000

Downloads.download("https://vincentarelbundock.github.io/Rdatasets/csv/Ecdat/Participation.csv", "participation.csv")

readlines("participation.csv")

df_raw = CSV.read("participation.csv", DataFrame)

describe(df_raw)

df = select(df_raw,
            :lfp => (x -> recode(x, "yes" => 1, "no" => 0)) => :lfp,
            :lnnlinc,
            :age,
            :age => ByRow(x -> x^2) => :age²,
            Between(:educ, :noc),
            :foreign => categorical => :foreign)

df = select(df_raw,
            :lfp => x -> recode(x, "yes" => 1, "no" => 0),
            :lnnlinc,
            :age,
            :age => ByRow(x -> x^2) => :age²,
            Between(:educ, :noc),
            :foreign => categorical,
            renamecols=false)

describe(df)

@chain df begin
    groupby(:lfp)
    combine([:lnnlinc, :age, :educ, :nyc, :noc] .=> mean)
end

[:lnnlinc, :age, :educ, :nyc, :noc] .=> mean

@chain df begin
    groupby(:lfp)
    combine(names(df, Real) .=> mean)
end

@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
end

@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
    unstack(:lfp, :foreign, :nrow)
end

@chain df begin
    groupby([:lfp, :foreign])
    combine(nrow)
    unstack(:lfp, :foreign, :nrow)
    select(:lfp, [:no, :yes] => ByRow((x, y) -> y / (x + y)) => :foreign_yes)
end

@chain df begin
    groupby(:lfp)
    combine(:foreign => (x -> mean(x .== "yes")) => :foreign_yes)
end

gd = groupby(df, :lfp)

gd[1]

gd[(lfp=0,)]

gd[Dict(:lfp => 0)]

gd[(0,)]

@df df density(:age, group=:lfp)

probit = glm(@formula(lfp ~ lnnlinc + age + age² + educ + nyc + noc + foreign),
             df, Binomial(), ProbitLink())

probit = glm(Term(:lfp) ~ sum(Term.(propertynames(df)[2:end])),
             df, Binomial(), ProbitLink())

Term(:lfp) ~ sum(Term.(propertynames(df)[2:end]))

@formula(lfp ~ lnnlinc + age + age² + educ + nyc + noc + foreign)

probit = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
             df, Binomial(), ProbitLink())

@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign)

df_pred = DataFrame(lnnlinc=10.0, age= 2.0:0.01:6.2, educ = 9, nyc = 0, noc = 1, foreign = "yes")

probit_pred = predict(probit, df_pred, interval=:confidence)

plot(df_pred.age, Matrix(probit_pred), labels=["lfp" "lower" "upper"],
     xlabel="age", ylabel="Pr(lfp=1)")

probit

function boot_sample(df)
    df_boot = df[rand(1:nrow(df), nrow(df)), :]
    probit_boot = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
                      df_boot, Binomial(), ProbitLink())
    return (; (Symbol.(coefnames(probit_boot)) .=> coef(probit_boot))...)
end

function run_boot(df, reps)
    coef_boot = DataFrame()
    for _ in 1:reps
        push!(coef_boot, boot_sample(df))
    end
    return coef_boot
end

Random.seed!(1234)
@time coef_boot = run_boot(df, 1000)

conf_boot = mapcols(x -> quantile(x, [0.025, 0.975]), coef_boot)

confint(probit)

conf_param = DataFrame(permutedims(confint(probit)), names(conf_boot))

append!(conf_boot, conf_param)

insertcols!(conf_boot, 1, :statistic => ["boot lo", "boot hi", "parametric lo", "parametric hi"])

conf_boot_t = permutedims(conf_boot, :statistic)

insertcols!(conf_boot_t, 2, :estimate => coef(probit))

select!(conf_boot_t, :statistic, :estimate, 3:6 .=> x -> abs.(x .- conf_boot_t.estimate), renamecols=false)

scatter(0.05 .+ (1:8), conf_boot_t.estimate,
        yerror=(conf_boot_t."boot lo", conf_boot_t."boot hi"),
        label="bootstrap",
        xticks=(1:8, conf_boot_t.statistic), xrotation=45)
scatter!(-0.05 .+ (1:8), conf_boot_t.estimate,
         yerror=(conf_boot_t."parametric lo", conf_boot_t."parametric hi"),
         label="parametric")

function boot_probit(df_boot)
    probit_boot = glm(@formula(lfp ~ lnnlinc + age + age^2 + educ + nyc + noc + foreign),
                      df_boot, Binomial(), ProbitLink())
    return (; (Symbol.(coefnames(probit_boot)) .=> coef(probit_boot))...)
end

Random.seed!(1234)
@time bs = bootstrap(boot_probit, df, BasicSampling(1000))

bs_ci = confint(bs, PercentileConfInt(0.95))

conf_boot_t.bootstrap = [(ci[1], ci[1] - ci[2], ci[3] - ci[1]) for ci in bs_ci]

conf_boot_t

select!(conf_boot_t, Not(:bootstrap), :bootstrap => ["estimate 2", "boot lo 2", "boot hi 2"])

select(conf_boot_t, :statistic, r"estimate", r"lo", r"hi")

sort(conf_boot_t, :estimate)

conf_boot_t[sortperm(conf_boot_t."boot hi" + conf_boot_t."boot lo"), :]
