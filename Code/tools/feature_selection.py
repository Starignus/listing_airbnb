#!/usr/bin/env/python

"""
Script for feature selection Genetic algorithm and utilities
"""

import numpy as np
from sklearn.model_selection import KFold
import random
import matplotlib.pyplot as plt
import pandas as pd


def initilisation_of_population(size, n_feat, percent_not_feat=0.3,
                                seed_on=True, seed=42):
    """
    Initialisig the population flags. Each chromosome has the flag for each
    feature or gene that forms the chromosome. The population is the number of
    chromosomes we take into account. The probability that each feature or gene
    appears in the chromosome is 1-percent_not_feat.

    :param size: integer. Size of smple or population
    :param n_feat: Number of features or genes that forms the chromosome
    :param percent_not_feat: percetage of population with False/ label. This
    means the feature or gene that is not in the chromosome. The porbability
    that a feature or gene is taken into account is 1-percent_not_feat.
    :return: list of 'size' as number of elements that represents the
    population. Each element is an array of 1 x n_feat dimensions that defines
    our chromosome. This array contains the flags for our features that forms
    the chromosome.
    """
    population = []
    # The seed is imposed for reproducibility
    if seed_on:
      np.random.seed(seed)
    for i in range(size):
        # initial array with ones
        chromosome = np.ones(n_feat, dtype=np.bool)
        # Percentage of the features with label false in the chromosome
        chromosome[:int(percent_not_feat * n_feat)] = False
        # Shuffles the array along the first axis of a multi-dimensional array.
        # Modify a sequence in-place by shuffling its contents.
        np.random.shuffle(chromosome)
        population.append(chromosome)
    return population


def fitness_score_v3(population, clf, x_train, y_train, metric, n_splits,
                     random_state, shuffle=True, classification=False):
  """
  Function that gets the scores and population after fitness. Find the
  fitness value of the each of the chromosomes(a chromosome is a set of
  features or genes which define a proposed solution to the problem that
  the genetic algorithm is trying to solve).
  The scores are sorted descending, as well as the chromosomes in the population
  so we know which are the stronger chromosomes in the population.

  :pram population: list with all the population of chromosomes
  :param clf: model instance
  :param x_train: pandas data frame with train sample.
  :param y_train: pandas series or numpy array with train target
  :param metric: function for metric to be used as scoring approach
  :param n_splits:
  :param random_state:
  :param shuffle:
  :param classification

  :return: list of the scores and population in descending order. Therefore,
  the first score and first chromosome in the population is the stronger one.
  """
  # Creating copies
  x_train_c = x_train.copy()
  y_train_c = y_train.copy()

  #  data partition validation
  kf = KFold(n_splits=n_splits, shuffle=shuffle,random_state=random_state)

  # list of the population (each score per chromosome)
  scores = []
  # A chromosome is an array with the flags for each feature
  # that indincates wich feature will be used when training
  for chromosome in population:
    #print(chromosome)
    #print(x_train.iloc[:, chromosome].columns)
    scores_0 = []
    for train_indexs, test_indexs in kf.split(x_train_c):
      x_train_split = x_train_c.iloc[train_indexs, :]
      x_test_split = x_train_c.iloc[test_indexs, :]
      y_train_split = y_train_c[train_indexs]
      y_test_split = y_train_c[test_indexs]

      clf.fit(x_train_split.iloc[:, chromosome], y_train_split)
      predictions = clf.predict(x_test_split.iloc[:, chromosome])
      scores_0.append(metric(y_test_split, predictions))
    scores.append(np.mean(scores_0))
  # arrays of the scores and chromosomes in our population
  scores, population = np.array(scores), np.array(population)
  # Getting the indexes of the scores sorted ascending
  score_idx_scd = np.argsort(scores)
  if classification:
    # We want the greater score at the top
    # Getting scores descending order in a list.
    # scores is an array population size X 1
    scores_dsc = list(scores[score_idx_scd][::-1])
    # Getting chromosome flags in descending order of the scores in a list.
    # popuation is an array of population size X number of features
    # [::-1] reverse the view
    population_dsc = list(population[score_idx_scd, :][::-1])
    return scores_dsc, population_dsc
  else:
    # We want for regression the smallest score or error
    scores_scd = list(scores[score_idx_scd])
    population_scd = list(population[score_idx_scd, :])
    return scores_scd, population_scd


def selection(population_after_fit, n_parents):
    """
    This is the selection function that selects the best fitted chromosomes
    as parents (we select the number of parents) to pass the features or genes
    to the next generation and create a new population.

    :param population_after_fit: list of chromosomes in the population with
    the first member is the stronger chromosome
    :pram n_parents: integer that indicates the number of parents we are
    considering to pass their genes.

    :return: A list of arrays of chromosomes that will pass their
    genes to next generation
    """

    return population_after_fit[:n_parents]


def crossover_half_point(population_after_selection):
    """
    Function that create new set of chromosome by combining the parents
    and add them to new population set. We are matting and getting an
    offspring.

    :param population_after_selection: A list of arrays of chromosomes
    that will pass their genes to next generation.

    :return: A list of arrays of chromosomes after matting (parents and children).
    The previous and next generation.
    """
    population_after_crossover = population_after_selection.copy()
    for chromosome_idx in range(len(population_after_selection)):
        # Child
        child = population_after_selection[chromosome_idx].copy()
        # The module is the residual from integer division to take the
        # index of the subsequent chromosome in the list within our population.
        # The crossover or matting will happen between chromosome_{i} and
        # chromosome_{i+1}.
        chromosome_partner_idx = (1 + chromosome_idx) % len(population_after_selection)
        # The point of crossover takes place at the center  where the half
        # of the offspring will have the first half of the genes from
        # chromosome_{i} and the second part from chromosome_{i+1}.
        child[len(child) // 2:] = population_after_selection[chromosome_partner_idx][len(child) // 2:]
        population_after_crossover.append(child)
    return population_after_crossover


def mutation(population_after_crossover, n_parents, mutation_rate=0.2):
    """
    This function performs the mutation which alters one or more gene flag or
    feature flag values in a chromosome in the crossover or offspring.
    Mutation helps in getting more diverse opportunity. The population obtained
    will be used in the next generation.

    :param population_after_crossover: A list of arrays of chromosomes
    after matting (parents and children). The previous and next generation.
    :pram n_parents: integer that indicates the number of parents we are
    considering to pass their genes.
    :param mutation_rate: float that indicates the probability of mutation of a
    gene or feature flag.

    :return: A list of arrays of chromosomes after mutation
    """
    # Adding to the next generation the parents
    population_next_generation = population_after_crossover[:n_parents].copy()
    # Selecting just the offspring where the mutation take place
    offspring = population_after_crossover[n_parents:].copy()
    # Going through each chromosome (indivudal) in the population
    for chromosome in offspring:
        chromosome = chromosome.copy()
        # Select which features or genes will be muteted ramdomly
        for idx_feat in range(len(chromosome)):
            # Take the decision of mutation (flip the value) of the gene or feature
            # flag based on the mutation rate and a random number between [0,1].
            rand = random.random()
            if rand < mutation_rate:
                chromosome[idx_feat] = not chromosome[idx_feat]
        population_next_generation.append(chromosome)
    return population_next_generation


def generations_v4(size, n_feat, metric, clf, n_parents, mutation_rate,
                   n_generations, X_train, y_train, n_splits=3,
                   percent_not_feat=0.3, seed_on=False, seed=42,
                   random_state_fit=42, classification=False,
                   early_stop_ths=0.01, early_stop_falg=False):
    """

    :param size: integer, generation size
    :param n_feat: number of fetures to chose on
    :param metric: object of the metric
    :param clf: object classifier
    :param n_parents: integer number of parents
    :param mutation_rate: float, percentage of mutation of chromosomes
    :param n_generations: iteger, number of gerations
    :param X_train: data frame with X train
    :param X_test: data frame with X test
    :param y_train: data frame with y train
    :param y_test: data frame with y test
    :param percent_not_feat: percetage of population with False/ label. This
    means the feature or gene that is not in the chromosome. The porbability
    that a feature or gene is taken into account is 1-percent_not_feat.
    :param seed_on: boolean, random seed for reproducibility
    :param seed: integer random state.
    :return:
    """
    best_chromo = []
    best_score = []
    # Generating the flags for the chromosome (features)
    # in a population (size)-(Samples of arrays with chormosomes)
    population_nextgen = initilisation_of_population(size,
                                                     n_feat,
                                                     percent_not_feat=percent_not_feat,
                                                     seed_on=seed_on,
                                                     seed=seed)
    # Creating a number of generations to see which one survives
    for i in range(n_generations):
        scores, pop_after_fit = fitness_score_v3(population=population_nextgen,
                                                 clf=clf,
                                                 x_train=X_train,
                                                 y_train=y_train,
                                                 metric=metric,
                                                 n_splits=n_splits,
                                                 random_state=random_state_fit,
                                                 classification=classification)
        print(scores[:2])
        pop_after_sel = selection(population_after_fit=pop_after_fit,
                                  n_parents=n_parents)
        pop_after_cross = crossover_half_point(pop_after_sel)
        # Next generation after mutation for the next round
        population_nextgen = mutation(population_after_crossover=pop_after_cross,
                                      n_parents=n_parents,
                                      mutation_rate=mutation_rate)
        # append the best chromosome and score
        best_chromo.append(pop_after_fit[0])
        best_score.append(scores[0])
        if early_stop_falg:
            if len(best_score) > 1 and best_score[i] - best_score[i - 1] < early_stop_ths:
                break

    return best_chromo, best_score


def best_in_gene(best_scores):
    plt.plot(best_scores)
    plt.xlabel("Iterations")
    plt.ylabel("Fitness metrics")


def bests_chromos_coparison(bests_chromos, best_scores):
    """
    Comparing the best chromosomes to look for degenerated classes.

    :param bests_chromos: list with the chromosomes as numpy arrays
    :param best_scores: list with the score of each chromosome

    :return: list of tuples with the comparison and score:
    (index, pair of index compared, result of comparison,
    score of chromosome in the index)
    """
    list_compare = []
    for chromosome_idx in range(len(bests_chromos)):
        next_idx = (1 + chromosome_idx)%len(bests_chromos)
        list_compare.append((chromosome_idx,
                             "c{}-{}".format(chromosome_idx, next_idx),
                            (bests_chromos[chromosome_idx] == bests_chromos[next_idx]). all(),
                            "sc-{}".format(chromosome_idx), best_scores[chromosome_idx],
                             "# features {}".format(np.sum(bests_chromos[chromosome_idx]))))
    return list_compare


def feature_importance(model, model_type, feature_list, plot=True):
    if model_type == 'Calibrated':
        contributions = np.sum([i.base_estimator.feature_importances_ for i in model.calibrated_classifiers_], axis=0)
        f_importance = np.array(contributions)
    else:
        f_importance = np.array(model.feature_importances_)
    df_f_importance = pd.DataFrame(f_importance,
                                   index=feature_list,
                                   columns=['Feature_importance%']).sort_values('Feature_importance%', ascending=True)
    df_f_importance = 100 * (df_f_importance / df_f_importance.sum())
    if plot:
        fig = plt.figure(figsize=(10, 20))
        plt.barh(df_f_importance.index, df_f_importance['Feature_importance%'], color='blue')
        plt.title('Feature Importance')
    return df_f_importance.sort_values('Feature_importance%', ascending=False)


def xgboost_learning_curve(data, train_label=None, test_label=None,
                           std_col=None, figsize=(10, 5),
                           y_lim=None, metric='RMSE'):
    if isinstance(data, pd.DataFrame):
        x_axis = range(0, data.shape[0])
        y_train = data[train_label]
        y_test = data[test_label]
        print(train_label, y_train.min())
        print(test_label, y_test.min())
        if std_col is not None:
            print(std_col[0], data[std_col[0]].min())
            print(std_col[1], data[std_col[1]].min())
    elif isinstance(data, dict):
        keys_val = list(data.keys())
        list_metric = list(data[keys_val[0]].keys())
        x_axis = range(0, len(data[keys_val[0]][list_metric[0]]))
        y_train = data[keys_val[0]][list_metric[0]]
        y_test = data[keys_val[1]][list_metric[0]]
    else:
        print("It should be a dictionary or data frame")
    # plot log loss
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(x_axis, y_train, label='Train')
    ax.plot(x_axis, y_test, label='Test')
    ax.legend()
    plt.ylabel(metric)
    plt.title('XGBoost Regression')
    plt.xlabel('Boosting trees')
    if y_lim is not None:
        plt.ylim(y_lim)
