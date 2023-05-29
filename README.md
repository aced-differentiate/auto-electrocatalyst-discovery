# A Multiobjective Closed-loop Approach Towards Autonomous Discovery of Electrocatalysts for Nitrogen Reduction

Data and scripts in support of the publication "A Multiobjective Closed-loop Approach Towards Autonomous Discovery of Electrocatalysts for Nitrogen Reduction", Kavalsky et al., (2023). DOI: [10.26434/chemrxiv-2023-vmbt3-v2](https://doi.org/10.26434/chemrxiv-2023-vmbt3-v2).

The repository is organized as follows:

1. [data/](data)

    * `acsl.json`: `autocat.learning.sequential.SequentialLearner` object containing all historical data from the sequential learning search. This may be read using the `SequentialLearner.from_json` method.

    * `acds.json`: `autocat.learning.sequential.DesignSpace` object containing all structures within the design space (with calculated labels where available). This may be read using the `DesignSpace.from_json` method.

    * `dft_data.db`: `ase.db` containing all of the generated DFT data from the search with entries in the [Physical Information File (PIF)](https://citrineinformatics.github.io/pif-documentation) format. This may be read using `ase.db.connect` using `type="json"`

    * `ELEMENTS.json`: json containing all chemical species considered in this study

    * `raw_volc_m_b.csv`: slopes and intercepts to reproduce the used activity volcano from "The challenge of electrochemical ammonia synthesis: a new perspective on the role of nitrogen scaling relations", Montoya et al., *ChemSusChem* **8** (13), 2180-2186 (2015). DOI: [10.1002/cssc.201500322](https://doi.org/10.1002/cssc.201500322)

    * [bee\_ensembles](data/bee_ensembles/)

        Text files with the BEE energy ensembles for each system that was autonomously identified during the search


2. [scripts/](scripts)

    * [aq\_hist\_plot](scripts/aq_hist_plot/)

        * `get_aq_hist.py`: Script for extracting the acquisition scores and prediction uncertainties as a function of sequential learning (SL) iteration into a text file

        * `make_aq_hist_plot.py`: Script to generate a plot of candidate acquisition scores and uncertainties against SL iteration.

        If these scripts are run as-is, will reproduce Figure 3b from the paper.

    * [drivers](scripts/drivers/)

        * `manage_dft_calculations.py`: Script for managing high-throughput adsorption energy calculations on a computing cluster using [`fireworks`](https://materialsproject.github.io/fireworks/). Will ensure that first the clean slabs are relaxed before placing the adsorbate.

        * `reference_energies.json`: Tabulated reference energies used to calculate $\Delta G_{\mathrm{N}}$ from the DFT total energies of the relaxed systems.

        * `sl_driver.py`: Script for driving the guided candidate selection with SL. Will automatically re-train the machine learning surrogate, re-calculate the acquisition scores, and suggest the next candidate system for evaluation.

    * [obj_space_hist_plot](scripts/obj_space_hist_plot/)

        * `extract_obj_space_hist.py`: Extracts the HHI, Segregation Energies, and $\Delta G_{\mathrm{N}}$ of both the systems in the initial training set as well as candidates as a function of SL iteration into text files.

        * `make_obj_space_hist_plot.py`: Script for generating two subplots. First, it will generate a subplot of the activity volcano with candidates. Second, it will generate a subplot of Normalized HHI against Segregation Energy. Both plots will have candidates colored based on SL iteration.

        If these scripts are run as-is, will reproduce Figure 4 from the paper.

    * [rank_score_plot](scripts/rank_score_plot/)

        * `get_ranking.py`: Calculates the partial scores ($c_j^{\mathrm{active}}$, $A_j$, $C_j$) and total ranking scores ($RS_j$) for all candidates and extracts the data into a text file

        * `make_ranking_plot.py`: Script for generating the ranking plot of the top 5 identified candidates

        If these scripts are run as-is, will reproduce Figure 5 from the paper

    * [umap\_plots](scripts/umap_plots/)

        * `L1_EMBEDDING.txt`: Contains the UMAP embeddings of all systems in the considered SAA design space that were used in the paper.

        * `make_umap_plot_initial_only`: Script for generating plot of UMAP projection with only the initial training points highlighted (Figure 1d in the paper)

        * `make_umap_plot.py`: Script for generating plot of UMAP projection with both the initial training points highlighted alongside the identified candidates as a function of iteration (Figure 3a in the paper)

        * `umap_calc.py`: Calculate UMAP embeddings for the SAA design space using magpie featurization. **N.B.** due to the stochasticity in the UMAP approach, running this script as-is does not guarantee identical embeddings to that provided in `L1_EMBEDDING.txt`, but overall trends should remain


## Running the scripts

The required packages for executing the scripts are specified in `requirements.txt`,
and can be installed in a new environment (e.g. using
[conda](https://docs.conda.io/projects/conda/en/latest/index.html))
as follows:

```py
$ conda create -n multi_obj_search python=3.10
$ conda activate multi_obj_search
$ pip install -r requirements.txt
```

The scripts are all in python, and can be run from the command line. For example:
```py
$ cd scripts/aq_hist_plot
$ python get_aq_hist.py
```
