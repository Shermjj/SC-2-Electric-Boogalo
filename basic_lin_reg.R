# Data Cleaning + Basic regression
library(ggplot2)
library(fpp2)
library(slider)
# Extract individual dataframes
load("~/docs/code/SC-2-Electric-Boogalo/data/Irish.RData")
indCons <- Irish[["indCons"]]
survey <- Irish[["survey"]]
extra <- Irish[["extra"]]

agg_fun <- function(df, agg_level){
  agg <- rowMeans(df)
  agg <- unlist(as.vector(agg), use.names = FALSE)
  
  agg_hour <- slide_dbl(agg[1:16798], ~sum(.x), .after = 1, .step=2)
  agg_hour <- na.omit(agg_hour)
  
  agg_day <- slide_dbl(agg_hour[1:8376], ~sum(.x), .after = 23, .step=24)
  agg_day <- na.omit(agg_day)
  if(agg_level == "day"){
    return(agg_day)
  } else if(agg_level == "hour"){
    return(agg_hour)
  } else if(agg_level == "30m"){
    return(agg)
  } else {
    return(NULL)
  }
}

soc_classes <- c(unique(survey["SOCIALCLASS"]))$SOCIALCLASS
cons_by_class <- list()
cons_by_class_hr <- list()
cons_by_class_day <- list()
cons_by_class_halfhr <- list()
agg_cols = 0

for(c in 1:length(soc_classes)){
  soc_class = as.character(soc_classes[c])
  id <- survey[,1][survey["SOCIALCLASS"]==soc_class] #Get ids of each social class
  cons_by_class[[soc_class]] = indCons[id]
  agg_cols = agg_cols + ncol(cons_by_class[[soc_class]])
  
  cons_by_class_halfhr[[soc_class]] = agg_fun(cons_by_class[[soc_class]], "30m")
  cons_by_class_hr[[soc_class]] = agg_fun(cons_by_class[[soc_class]], "hour")
  cons_by_class_day[[soc_class]] = agg_fun(cons_by_class[[soc_class]], "day")
}
agg_cols == ncol(indCons) #check

df_day <- data.frame(cons_by_class_day)
df_hr <- data.frame(cons_by_class_hr)
df_hfhr <- data.frame(cons_by_class_halfhr)
#Create different list of aggregation


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
extra_day <- extra[seq(1,16798,48),c("dow")] #16751 Why did wed skip to sun for the first index? some of the days skipped as well.....
extra_hr <- extra[seq(1,16798,24),c("dow")] #Why did wed skip to sun for the first index? some of the days skipped as well.....

df_hfhr <- cbind(df_hfhr, extra)
df_day$temp <- temp_day
df_day$dow <- extra_day
df_hr$temp <- temp_hour
df_hr$dow <- extra_hr

# Aggregate total
# Freqeuncy is 30 minutes, so each day has 48 ticks
autoplot(ts(agg[1:48*7]))
autoplot(ts(agg[1:48]))
# Aggregate to 1 hour freq., remove last element to ensure even
autoplot(ts(agg_fun(soc_cons$DE, "hour")[1:24*3])) + xlab("Hour") + ylab("Total Electricity Used")
# Aggregate to 1 day freq, remove last element to ensure even 
# 349 * 24 = 8376
autoplot(ts(agg_day)) + xlab("Day") + ylab("Total Electricity Used")

#saveRDS(df, "./data/daily_agg_data.RDS")

#Try lin. reg
lm(ele_con ~ dow + temp, df)

