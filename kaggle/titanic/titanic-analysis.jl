using CSV;
using DataFrames;
using Plots;


myData = CSV.File("./kaggle/titanic/gender_submission.csv") |> DataFrame

myData

describe(myData)

sum(myData[!,2])

plot(myData[!,2])

plot(rand(100))