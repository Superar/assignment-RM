library(stringr)

df <- read.csv("results-milestone3/results_12_31_2021_14_08_35.csv")
clean_info <- gsub("^data_", "", df$file)
clean_info <- gsub(".in$", "", clean_info)
split_info <- str_split_fixed(clean_info, "-", 3)

df[c("n_exams", "prob", "seed")] <- split_info
df$n_exams <- as.numeric((df$n_exams))
df$prob <- as.numeric((df$prob))
df$seed <- as.numeric((df$seed))

anova_result <- aov(log(runtime) ~ C(algorithm) * n_exams * prob, data = df)
print(summary(anova_result))
