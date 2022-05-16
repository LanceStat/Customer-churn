library(tidyverse)
library(VIM)
library(tidymodels)

train <- read_csv("~/Desktop/sliceds01e07/train.csv")
test <- read_csv("~/Desktop/sliceds01e07/test.csv")

#aggr(train)

na_sum <- train %>% 
  select(everything()) %>% 
  summarise_all(funs(sum(is.na(.))))

trainid <- train %>% 
  select(id)
  
train <- train %>% 
  mutate_at(vars(attrition_flag, gender, education_level, income_category), factor) %>% 
  select(-id)

testid <- test %>% 
  select(id)

test <- test %>% 
  mutate_at(vars(gender, education_level, income_category), factor) %>% 
  select(-id)

# split training data
set.seed(111)
split <- train %>% 
  initial_split(prop = 0.7, strata = attrition_flag)
cvtrain <- training(split)
cvtest <- testing(split)

# k-fold CV
set.seed(122)
folds <- cvtrain %>% 
  vfold_cv(v = 5, strata = attrition_flag)


xg_spec <- boost_tree(trees = tune(),
                      tree_depth = tune(),
                      learn_rate = tune(),
                      min_n = tune(),
                      sample_size = tune(),
                      mtry = tune()) %>% 
  set_engine("xgboost") %>% 
  set_mode("classification")

xg_rec <- recipe(attrition_flag ~ ., data = cvtrain) %>% 
  step_dummy(all_nominal_predictors(), one_hot = TRUE)

xg_wf <- workflow() %>% 
  add_model(xg_spec) %>% 
  add_recipe(xg_rec)

xg_grid <- grid_latin_hypercube(
  trees(range = c(50, 500)),
  tree_depth(range = c(4,8)),
  min_n(range = c(3, 14)),
  sample_size = sample_prop(),
  finalize(mtry(range = c(3, 14)), cvtrain),
  learn_rate(range = c(0.005, 0.02), trans = NULL),
  size = 50
)

grid_control <- control_grid(save_pred = TRUE,
                             save_workflow = TRUE,
                             extract = extract_model)

doParallel::registerDoParallel(cores = 5)
set.seed(123)
xg_tune <- xg_wf %>% 
  tune_grid(resamples = folds,
            metrics = metric_set(roc_auc),
            control = grid_control,
            grid = xg_grid)

write_rds(xg_tune, "~/Desktop/sliceds01e07/xg_tune_final.rds")

xg_tune %>% 
  autoplot()

xg_tune %>% 
  collect_metrics() %>% 
  arrange(-mean) %>% 
  slice_head(n = 1)

xg_best <- xg_tune %>% 
  select_best("roc_auc")

xg_final <- xg_wf %>% 
  finalize_workflow(xg_best)

xg_final %>% 
  fit(data = cvtrain) %>% 
  pull_workflow_fit() %>% 
  vip(geom = "point")

xg_last <- xg_final %>% 
  last_fit(split)

xg_last %>% 
  collect_predictions() %>% 
  roc_curve(attrition_flag, .pred_1, event_level = "second") %>% 
  ggplot(aes(1- specificity, sensitivity)) +
  geom_line(size = 1.5, color = "darkgreen") +
  geom_abline(lty = 2) +
  geom_path() +
  coord_fixed()

xg_last %>% 
  collect_predictions() %>% 
  roc_auc(attrition_flag, .pred_1, event_level = "second")

xg_fit_final <- xg_final %>%
  fit(train)

xg_pred <- predict(xg_fit_final, new_data = test, type = "prob")

xg_output <- tibble(testid$id, xg_pred$.pred_1)
colnames(xg_output) <- c("id", "attrition_flag")
write_csv(xg_output, "~/Desktop/sliceds01e07/xgsubmit.csv")
