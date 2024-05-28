library(xts)
library(electBook)

data("Irish")
indCons <- Irish[["indCons"]]
survey <- Irish[["survey"]]
extra <- Irish[["extra"]]
extra <- extra[49:16799,] #Start from 2010
row.names(extra) <- NULL
indCons <- indCons[49:16799,] #Start from 2010
row.names(indCons) <- NULL
soc_classes <- c(unique(survey["SOCIALCLASS"]))$SOCIALCLASS
cons_by_class <- list()
cons_by_class_halfhr <- list()
agg_cols = 0

for(c in 1:length(soc_classes)){
  soc_class = as.character(soc_classes[c])
  id <- survey[,1][survey["SOCIALCLASS"]==soc_class] #Get ids of each social class
  cons_by_class[[soc_class]] = indCons[id]
  #agg_cols = agg_cols + ncol(cons_by_class[[soc_class]])
  
  agg <- rowMeans(cons_by_class[[soc_class]])
  agg <- unlist(as.vector(agg), use.names = FALSE)
  
  cons_by_class_halfhr[[soc_class]] = agg
}
#agg_cols == ncol(indCons) #check

extra_df <- extra[c('temp','toy')]
df_halfhr = data.frame(cons_by_class_halfhr,extra_df)

df_halfhr = xts(df_halfhr, order.by = extra$dateTime)
df_hr = period.apply(df_halfhr, endpoints(df_halfhr, "hours"), mean)
df_day = period.apply(df_halfhr, endpoints(df_halfhr, "days"), mean)

#### Half hour ####
# Generate a complete time sequence
complete_halfhr <- seq(from = min(index(df_halfhr)), to = max(index(df_halfhr)), by = "30 min")
# Merge with the complete time sequence
df_halfhr_complete <- merge(df_halfhr, xts(order.by = complete_halfhr), all = TRUE)
# Add a counter column
df_halfhr_complete$counter <- 1:nrow(df_halfhr_complete)
halfhr_counter <- as.vector(df_halfhr_complete$counter[index(df_halfhr_complete) %in% index(df_halfhr)])

dow_halfhr = weekdays(index(df_halfhr))
df_halfhr = as.data.frame(df_halfhr)
df_halfhr$dow = as.factor(dow_halfhr)
df_halfhr$counter = halfhr_counter
cols <- names(df_halfhr)[1:7]
df_halfhr[cols] <- lapply(df_halfhr[cols], as.numeric)
saveRDS(df_halfhr, here::here("data/df_halfhr.RData"))

#### Hour ####
# Generate a complete time sequence
complete_hr <- seq(from = min(index(df_hr)), to = max(index(df_hr)), by = "1 hour")
# Merge with the complete time sequence
df_hr_complete <- merge(df_hr, xts(order.by = complete_hr), all = TRUE)
# Add a counter column
df_hr_complete$counter <- 1:nrow(df_hr_complete)
hr_counter <- as.vector(df_hr_complete$counter[index(df_hr_complete) %in% index(df_hr)])

dow_hr = weekdays(index(df_hr))
df_hr = as.data.frame(df_hr)
df_hr$dow = as.factor(dow_hr)
df_hr$counter = hr_counter
cols <- names(df_hr)[1:7]
df_hr[cols] <- lapply(df_hr[cols], as.numeric)
saveRDS(df_hr, here::here("data/df_hr.RData"))

#### Day ####
# Generate a complete time sequence
complete_day <- seq(from = min(index(df_day)), to = max(index(df_day)), by = "1 DSTday")
# Merge with the complete time sequence
df_day_complete <- merge(df_day, xts(order.by = complete_day), all = TRUE)
# Add a counter column
df_day_complete$counter <- 1:nrow(df_day_complete)
day_counter <- as.vector(df_day_complete$counter[index(df_day_complete) %in% index(df_day)])

dow_day = weekdays(index(df_day))
df_day = as.data.frame(df_day)
df_day$dow = as.factor(dow_day)
df_day$counter = day_counter
cols <- names(df_day)[1:7]
df_day[cols] <- lapply(df_day[cols], as.numeric)
saveRDS(df_day, here::here("data/df_day.RData"))
