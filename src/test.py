from game import Game
from simulator import Simulator

import argparse
import os

PATH = os.path.dirname(__file__)
DEFAULT_FIGURE_PATH = os.path.join(PATH, '../figures')
DEFAULT_SAVE_PATH = os.path.join(PATH, '../save')
DEFAULT_SAVE_FILE = 'save.txt'

parser = argparse.ArgumentParser(description='Script launch MC simulations')

parser.add_argument('--fig_path', type=str, default=DEFAULT_FIGURE_PATH, help='The folder path to save figures')
parser.add_argument('--save_path', type=str, default=DEFAULT_SAVE_PATH, help='The folder path to save results')
parser.add_argument('--save_file', type=str, default=DEFAULT_SAVE_FILE, help='The file in which append new results')
parser.add_argument('--nsim', type=int, default=1, help='Number of simulations for each MC algorithm')
parser.add_argument('--target', type=int, default=2048, help='The target tile to reach to win')
parser.add_argument('--nested', type=str, default='', help='A list of 0 and 1 separated by commas indicating for each test if we use nested (1) or flat (0)')
parser.add_argument('--params', type=str, default='', help='The levels/playouts for each test separated by commas')
parser.add_argument('--eval', type=str, default='', help='Indicates the value used for the evaluation function for each test separated by commas')
parser.add_argument('--transposition', type=str, default='', help='The threshold value for the transposition table for each test (no table if 0)')

args = parser.parse_args()

figure_path = args.fig_path
if not os.path.isdir(figure_path) : os.mkdir(figure_path)
save_path = args.save_path
if not os.path.isdir(save_path) : os.mkdir(save_path)
save_file=args.save_file
nsim = args.nsim
target = args.target
if len(args.nested)==0:
	nested = []
else:
	nested = [int(x) for x in args.nested.split(',')]
if len(args.params)==0:
	params = []
else:
	params = [int(x) for x in args.params.split(',')]
if len(args.eval)==0:
	eval = []
else:
	eval = [int(x) for x in args.eval.split(',')]
if len(args.transposition)==0:
	transposition = []
else:
	transposition = [int(x) for x in args.transposition.split(',')]

if __name__ == '__main__':

	if len(params)!=len(nested) or len(eval)!=len(nested) or len(transposition)!=len(nested):
		print('Error: --nested, --params, --eval and --transposition should contain the same number of parameters (separated by commas).')
		exit(0)

	sim = Simulator(target, nested, params, eval, transposition)
	sim.simulate(nsim)

	title = 'wins_nested{}_params{}_eval{}_transpo{}_nsims{}.png'.format(args.nested, args.params, args.eval, args.transposition, nsim)
	sim.plot_wins(os.path.join(figure_path, title))

	title = 'reaches_nested{}_params{}_eval{}_transpo{}_nsims{}.png'.format(args.nested, args.params, args.eval, args.transposition, nsim)
	sim.plot_reaches(os.path.join(figure_path, title))

	sim.save_results(os.path.join(save_path, save_file))
