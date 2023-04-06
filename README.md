# Multiple_shortest_path_problem_with_path_deconfliction

Project for the "Mathematical Optimization 2020/21" course at University of Trieste. This project is based on [Hughes, Lunday, Weir & Hopkinson, 2021](https://github.com/kkevin98/Multiple_shortest_path_problem_with_path_deconfliction/blob/main/paper.pdf) article.


## Project environment

To create the conda environment with the required packages to run noteboooks:

    conda env create -n CHOOSE_ENV_NAME --file environment.yml


## Structure

In `/data` folder are stored the same instances used by the authors within the article and are also used in this project. The unique difference is that they have been converted to `.csv` format for ease of use. Each file in the folder contains one or more network instances composed of their arcs with relative weights. The originals can be found at https://github.com/mike6hughes/Multiple-Shortest-Path-Problem-with-Path-Deconfliction.  
The `/utils` folder contains some useful modules, used inside the notebooks, to read and retrieve the datas of the previous folder, define the article's models and plot the results. The formulation of the different models can be found in [`problem_model.py`](https://github.com/kkevin98/Multiple_shortest_path_problem_with_path_deconfliction/blob/main/utils/problem_model.py).   
In `/data` are stored the results of the analysis that require a lot of computational times to be perfomed. They can be read instead of waiting the analysis' results each time.  

## Usage

To follow the same workflow used inside the article is suggested to use the notebooks in the following order:

1. `MSPP_implementation.ipynb`

    It contains the formulation and an application example of the **Multiple Shortest Path Problem (MSPP)**. Refers to section 3.2 of the article.

2. `MSPP-PD(ABP)_implementation.ipynb`

    It contains the formulation and an application example of the **Multiple Shortest Path Problem with Path Deconfliction(MSPP-PD)** in case 
    of arc binary penalties (**ABP**).  
    It also points-out the non-uniqueness of the solution to this kind of problem. Refers to section 3.2 of the article.

3. `MSPP-PD(NQP)_implementation.ipynb`

    It contains the formulation and an application example of the **Multiple Shortest Path Problem with Path Deconfliction(MSPP-PD)** in case 
    of node quadratic penalties (**NQP**).  
    Furthermore it explore the pareto solutions of the problem in case of a decision-maker that does not know a propri the relative importances of distances covered and penalty incurred by agents that have to be routed. Refers to section 3.2 of the article.

4. `Model_comparison.ipynb`

    It shows how increasing the number of agents traversing the network affects the optimal solutions to the MSPP-PD variants. Refers to section 3.3 of the article.

5. `Model_practability.ipynb`

    It analyze the practical tractability (*practability*) of the differents MSPP-PD variants. Refers to section 3.4 of the article.

6. `extra.ipynb`

    It first investigate how the symmetry between agents' sources and termini affect the practability of the MSPP-PD(ABP) and then present a possible real case application of the same model.

The first 5 notebooks reproduce the examples and the results that can be found in the article. The contents of the latter, on the other hand, are not found in the article.  
Also notice that, in the noteboks' tests, the instances used are a subset of those used in the article for computational times reason.
