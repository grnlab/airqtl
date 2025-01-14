#!/usr/bin/python3
# Copyright 2025, Lingfei Wang
#
# This file is part of airqtl.

__all__ = ['air','association', 'cov', 'heritability', 'io', 'kinship', 'op', 'pipeline', 'sim', 'utils']

from . import *


def _main_func_parser(parser,funcs):
	parser.add_argument('-v',
				dest='verbose',
				action='store_true',
				help='Verbose mode.')
	return parser,funcs

def _main_func_args(args):
	import logging
	import sys
	logging.basicConfig(
		format=
		'%(levelname)s:%(process)d:%(asctime)s:%(pathname)s:%(lineno)d:%(message)s',
		level=logging.DEBUG if args.verbose else logging.WARNING)
	logging.info('Started: '+' '.join([f"'{x}'" for x in sys.argv]))
	return args

def _main_func_ret(args,ret):
	import logging
	import sys
	logging.info('Completed: '+' '.join([f"'{x}'" for x in sys.argv]))
	return ret

def main():
	import docstring2argparse as d
	d.docstringrunner('airqtl.pipeline',func_filter=lambda name,obj:name[0]!='_' and hasattr(obj,'_da') and obj._da==True,func_parser=_main_func_parser,func_args=_main_func_args,func_ret=_main_func_ret)

assert __name__ != "__main__"
