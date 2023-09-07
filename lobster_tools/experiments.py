# AUTOGENERATED! DO NOT EDIT! File to edit: ../notebooks/05_experiments.ipynb.

# %% auto 0
__all__ = ['CONTEXT_SETTINGS', 'testa', 'cli_example']

# %% ../notebooks/05_experiments.ipynb 4
import click

# %% ../notebooks/05_experiments.ipynb 8
# simple test of click and setuptools console_scripts entry points.. works nicely for simple CLIs
CONTEXT_SETTINGS = dict(help_option_names=['-h', '--help'],
                        token_normalize_func=lambda x: x.lower() if isinstance(x, str) else x # can run with --COUNT or --count, --duck or --dUcK etc..
                        )

@click.command(context_settings=CONTEXT_SETTINGS)
@click.option('-c', '--count' , required=True, help='example arg.')
@click.option('-i', '--int' , 'int_', default=2, show_default=True, help='example arg.')
@click.option('-d', '--duck', is_flag=True, show_default=True, help='some flag.')
@click.option('-e', '--elephant', is_flag=True, show_default=True, default=True, help='some flag.')
def testa(count, duck, elephant, int_):
    "nbdev cli test with click"
    print(f"printing arg count: {count}. duck: {duck}. elephant: {elephant}. reserved keyword int: {int_}")

# %% ../notebooks/05_experiments.ipynb 9
def cli_example():
    "nbdev cli test"
    print("example_of_cli")
