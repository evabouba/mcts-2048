from game import Game
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import time
from joblib import Parallel, delayed, cpu_count

class Simulator:
  def __init__(self, target, nested, params, eval, transposition):
    self.target = target
    self.nested = nested
    self.params = params
    self.eval = eval
    self.transposition = transposition
    self.wins = []
    self.reaches = []
    self.scores = []
    self.moves = []

  def play(self, i):
    #print(i)
    game = Game(self.target, eval=self.eval[i], transposition=self.transposition[i])
    if self.nested[i]:
      score = game.NMCS(self.params[i])
    else:
      score = game.MCS(self.params[i])

    return [score, np.max(game.board), game.won, game.moves]

  def simulate(self, n):
    for i in range(len(self.nested)):
      start = time.time()
      if self.nested[i]:
        print("Nested", end=' ')
      else:
        print("Flat", end=' ')
      print('{} eval {} transposition {}'.format(self.params[i], self.eval[i], self.transposition[i]))
      
      results = np.array(Parallel(n_jobs=min(n, cpu_count()))(delayed(self.play)(i) for j in range(n)))
      print("Time ", time.time()-start)
  
      self.scores.append(results[:,0])
      self.reaches.append(results[:,1])
      self.wins.append(results[:,2])
      self.moves.append(results[:,3])


  def plot_wins(self, save_path=None):
    # wins : liste 2D de taille nb_param x n_simulations indiquant pour chaque partie si elle est gagnée ou pas
    # params : liste de level/n_playouts
    # nested : liste de booleen
    label = np.where(self.nested, "Nested", "Flat")
    plt.bar([str(label[i])+str(self.params[i])+' e{} t{}'.format(self.eval[i], self.transposition[i]) for i in range(len(self.nested))], np.mean(self.wins, axis=1), color=sns.color_palette()[:len(self.params)])
    plt.title("Winning rate")
    plt.ylim((0,1))
    if not save_path is None:
      plt.savefig(save_path)
 
 
  def plot_reaches(self, save_path=None):
    # reaches : liste 2D de taille nb_param x n_simulations indiquant pour chaque partie la valeur max atteinte
    # params : liste de level/n_playouts
    # nested : liste de booleen
    df = pd.DataFrame(columns=["param", "reaches", "Max Value"])
    label = np.where(self.nested, "Nested", "Flat")
    values = sorted(np.unique(self.reaches))

    for i in range(len(self.params))[::-1]:
      for reach in values[::-1]:

        df = pd.concat([pd.DataFrame([[str(label[i])+" "+str(self.params[i])+' e{} t{}'.format(self.eval[i], self.transposition[i]), np.mean(self.reaches[i]==reach), reach]], columns=["param", "reaches", "Max Value"]), df])
 
    plt.figure()
    sns.barplot(data=df, x="Max Value", y="reaches", hue="param")
    plt.title("Reaches")
    plt.legend(loc='upper left')
    if not save_path is None:
      plt.savefig(save_path)

  def save_results(self, save_path=None):
    if save_path is None:
      print("No save path given, not saving results.")
      return
    f = open(save_path, 'a')
    f.write("target {}\n".format(self.target))
    f.write("nested \t {}\n".format(self.nested))
    f.write("params \t {}\n".format(self.params))
    f.write("eval \t {}\n".format(self.eval))
    f.write("transp.\t {}\n".format(self.transposition))
    f.write("wins \t {}\n".format([list(x) for x in self.wins]))
    f.write("reaches\t {}\n".format([list(x) for x in self.reaches]))
    f.write("scores \t {}\n".format([list(x) for x in self.scores]))
    f.write("moves \t {}\n\n".format([list(x) for x in self.moves]))
    f.close()
