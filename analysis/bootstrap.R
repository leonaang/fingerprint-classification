library(readxl)
library(boot)
scanner <- read_excel("C:/Users/leona/OneDrive/DSA4199/NIST-Classification/TEST/results.xlsx", sheet = 1, col_names = TRUE)
magnetic <- read_excel("C:/Users/leona/OneDrive/DSA4199/NIST-Classification/TEST/results.xlsx", sheet = 2, col_names = TRUE)
brown <- read_excel("C:/Users/leona/OneDrive/DSA4199/NIST-Classification/TEST/results.xlsx", sheet = 3, col_names = TRUE)
joss <- read_excel("C:/Users/leona/OneDrive/DSA4199/NIST-Classification/TEST/results.xlsx", sheet = 4, col_names = TRUE)
inkpad <- read_excel("C:/Users/leona/OneDrive/DSA4199/NIST-Classification/TEST/results.xlsx", sheet = 5, col_names = TRUE)

# Create a function to take a resample of the values, and then calculate the mean 5-class accuracy
boot_mean <- function(original_vector, resample_vector) {
  mean(original_vector$Match[resample_vector])
}

set.seed(42)
# R is number of replications
mean_results_scanner <- boot(scanner, boot_mean, R = 10000)
boot.ci(mean_results_scanner)
mean_results_scanner
mean(mean_results_scanner$t)

mean_results_magnetic <- boot(magnetic, boot_mean, R = 10000)
boot.ci(mean_results_magnetic)
mean_results_magnetic
mean(mean_results_magnetic$t)

mean_results_brown <- boot(brown, boot_mean, R = 10000)
boot.ci(mean_results_brown)
mean_results_brown
mean(mean_results_brown$t)

mean_results_joss <- boot(joss, boot_mean, R = 10000)
boot.ci(mean_results_joss)
mean_results_joss
mean(mean_results_joss$t)

mean_results_inkpad <- boot(inkpad, boot_mean, R = 10000)
boot.ci(mean_results_inkpad)
mean_results_inkpad
mean(mean_results_inkpad$t)

# Create a function to take a resample of the values, and then calculate the mean 3-class accuracy
boot_mean <- function(original_vector, resample_vector) {
  mean(original_vector$Match2[resample_vector])
}

set.seed(42)
# R is number of replications
mean_results_scanner <- boot(scanner, boot_mean, R = 10000)
boot.ci(mean_results_scanner)
mean_results_scanner
mean(mean_results_scanner$t)

mean_results_magnetic <- boot(magnetic, boot_mean, R = 10000)
boot.ci(mean_results_magnetic)
mean_results_magnetic
mean(mean_results_magnetic$t)

mean_results_brown <- boot(brown, boot_mean, R = 10000)
boot.ci(mean_results_brown)
mean_results_brown
mean(mean_results_brown$t)

mean_results_joss <- boot(joss, boot_mean, R = 10000)
boot.ci(mean_results_joss)
mean_results_joss
mean(mean_results_joss$t)

mean_results_inkpad <- boot(inkpad, boot_mean, R = 10000)
boot.ci(mean_results_inkpad)
mean_results_inkpad
mean(mean_results_inkpad$t)
