# Data Cleaning + Basic regression
library(ggplot2)
library(fpp2)
library(slider)

# Extract individual dataframes
load("~/data/Irish.RData")
indCons <- Irish[["indCons"]]
survey <- Irish[["survey"]]
extra <- Irish[["extra"]]

# Aggregate total
# Freqeuncy is 30 minutes, so each day has 48 ticks
agg <- rowSums(indCons)
autoplot(ts(agg[1:48*7]))
autoplot(ts(agg[1:48]))
agg <- unlist(as.vector(agg), use.names = FALSE)
# Aggregate to 1 hour freq., remove last element to ensure even
agg_hour <- slide_dbl(agg[1:16798], ~sum(.x), .after = 1, .step=2)
agg_hour <- na.omit(agg_hour)
autoplot(ts(agg_hour[1:24*3])) + xlab("Hour") + ylab("Total Electricity Used")
# Aggregate to 1 day freq, remove last element to ensure even 
# 349 * 24 = 8376
agg_day <- slide_dbl(agg_hour[1:8376], ~sum(.x), .after = 23, .step=24)
agg_day <- na.omit(agg_day)
autoplot(ts(agg_day)) + xlab("Day") + ylab("Total Electricity Used")

# Work with daily frequency for now
# Deal with `extra`, keep only dow and temp (any more?) (holy is useless since no true holy? why all FALSE?)
extra <- extra[c('dow', 'temp')]
extra$dow <- as.factor(extra$dow)
temp <- extra$temp
temp_hour <- slide_dbl(temp[1:16798], ~mean(.x), .after = 1, .step=2)
temp_hour <- na.omit(temp_hour)
temp_day <- slide_dbl(temp_hour[1:8376], ~mean(.x), .after = 23, .step=24)
temp_day <- na.omit(temp_day)
autoplot(ts(temp_day)) + xlab("Day") + ylab("Mean Daily Temp. ")
extra_day <- extra[seq(1,16751,48),c("dow")] #Why did wed skip to sun for the first index? some of the days skipped as well.....
df <- data.frame(agg_day, extra_day, temp_day)
names(df) <- c("ele_con", "dow", "temp")
#saveRDS(df, "./data/daily_agg_data.RDS")

#Try lin. reg
lm(ele_con ~ dow + temp, df)

