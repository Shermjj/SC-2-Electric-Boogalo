library(forecast)
library(ElecForecast)
df_day <- readRDS("./data/df_day.RData")
df_halfhr <- readRDS("./data/df_halfhr.RData")
df_hr <- readRDS("./data/df_hr.RData")
autoplot(ts(df_day[,1]))

mod <- linReg(df_day)
         
summary(mod)
