library(dplyr)
library(tibble)
library(lubridate)

# Load data
df_halfhr <- readRDS(here::here("data/df_halfhr.RData"))
df_hr <- readRDS(here::here("data/df_hr.RData"))
df_day <- readRDS(here::here("data/df_day.RData"))

# Feature transformations
#### Half hour ####
df_halfhr_features <- df_halfhr %>% 
  # Rownames to column
  rownames_to_column("dateTime") %>% 
  mutate(dateTime = as_datetime(dateTime)) %>% 
  # Feature engineering
  mutate(
    hour = hour(dateTime),
    month = month(dateTime)
  ) %>% 
  mutate(temp_sq = temp^2) %>%  # quadratic term for temperature
  # One-hot encode the day of the week
  bind_cols(model.matrix(~ dow - 1, data = .)) %>%
  select(-dow)
# Scaled
df_halfhr_scaled <- df_halfhr_features %>% 
  mutate(across(-c(dateTime, counter), scale))
saveRDS(df_halfhr_scaled, here::here("data/df_halfhr_scaled.RData"))

#### Hour ####
df_hr_features <- df_hr %>% 
  # Rownames to column
  rownames_to_column("dateTime") %>% 
  mutate(dateTime = as_datetime(dateTime)) %>% 
  # Feature engineering
  mutate(
    hour = hour(dateTime),
    month = month(dateTime)
  ) %>% 
  mutate(temp_sq = temp^2) %>%  # quadratic term for temperature
  # One-hot encode the day of the week
  bind_cols(model.matrix(~ dow - 1, data = .)) %>%
  select(-dow)
# Scaled
df_hr_scaled <- df_hr_features %>% 
  mutate(across(-c(dateTime, counter), scale))
saveRDS(df_hr_scaled, here::here("data/df_hr_scaled.RData"))

#### Day ####
df_day_features <- df_day %>% 
  # Rownames to column
  rownames_to_column("dateTime") %>% 
  mutate(dateTime = as_datetime(dateTime)) %>% 
  # Feature engineering
  mutate(
    # hour = hour(dateTime), # No longer makes sense to do hour
    month = month(dateTime)
  ) %>% 
  mutate(temp_sq = temp^2) %>%  # quadratic term for temperature
  # One-hot encode the day of the week
  bind_cols(model.matrix(~ dow - 1, data = .)) %>%
  select(-dow)
# Scaled
df_day_scaled <- df_day_features %>% 
  mutate(across(-c(dateTime, counter), scale))
saveRDS(df_day_scaled, here::here("data/df_day_scaled.RData"))

