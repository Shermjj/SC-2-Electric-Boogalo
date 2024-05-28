library(xts)

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
df_halfhr$dow = weekdays(index(df_halfhr))
df_halfhr = as.data.frame(df_halfhr)
df_halfhr$dow = as.factor(df_halfhr$dow)
cols <- names(df_halfhr)[1:7]
df_halfhr[cols] <- lapply(df_halfhr[cols], as.numeric)
saveRDS(df_halfhr, "df_halfhr.RData")

df_hr$dow = weekdays(index(df_hr))
df_hr = as.data.frame(df_hr)
df_hr$dow = as.factor(df_hr$dow)
cols <- names(df_hr)[1:7]
df_hr[cols] <- lapply(df_hr[cols], as.numeric)
saveRDS(df_hr, "df_hr.RData")

df_day$dow = weekdays(index(df_day))
df_day = as.data.frame(df_day)
df_day$dow = as.factor(df_day$dow)
cols <- names(df_day)[1:7]
df_day[cols] <- lapply(df_day[cols], as.numeric)
saveRDS(df_day, "df_day.RData")
