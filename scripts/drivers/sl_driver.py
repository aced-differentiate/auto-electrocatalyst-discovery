"""
Implementation of an automated sequential learning (SL) driver for catalyst
search within the single-atom alloy (SAA) design space.

The following workflow is implemented:
1. The full design space is defined as 30C2 (over transition metal elements)
2. The current DesignSpace object is loaded from disk, deserialized
3. Current dataset from the ASE DB is queried and a new DesignSpace
object is initialized
4. Next iteration of SL is initiated, and candidates selected

"""
import os
import json
import logging
import itertools as it
from collections import Counter
from typing import List
from typing import Callable

import numpy as np
from sklearn.ensemble import RandomForestRegressor as SklearnRandomForestRegressor

import ase
import ase.db
from ase.data import reference_states
from ase.data import atomic_numbers

from pypif.pif import System
from pypif import pif

from matminer.featurizers.composition import ElementProperty

from autocat.saa import generate_saa_structures
from autocat.learning.sequential import DesignSpace
from autocat.learning.sequential import SequentialLearner
from autocat.learning.sequential import CandidateSelector
from autocat.learning.featurizers import Featurizer
from autocat.learning.predictors import Predictor


# Module level variables with logging 
LOG_FILE = os.path.join(os.path.expanduser("~"), "sequential.log")
logging.basicConfig(filename=LOG_FILE, level=logging.DEBUG)


# list of elements in the design space
thisdir = os.path.dirname(os.path.abspath(__file__))
# N.B. if running this script elsewhere, make sure to update this PATH
ELEMENTS_FILE = os.path.join(thisdir, "..", "..", "data", "ELEMENTS.json")
with open(ELEMENTS_FILE, "r") as fr:
    ELEMENTS = json.load(fr)


class RandomForestRegressor(SklearnRandomForestRegressor):
    """Custom wrapper for sklearn RandomForest regressor which returns random
    forest predictions along with std in the tree estimates."""

    def predict(self, X, return_std=True):
        preds = SklearnRandomForestRegressor.predict(self, X)
        est_preds = np.empty((len(X), len(self.estimators_)))
        # loop over each tree in the forest and use it to make a prediction
        for ind, est in enumerate(self.estimators_):
            est_preds[:, ind] = est.predict(X)
        # assert np.allclose(np.mean(est_preds, axis=1), preds)
        if return_std:
            return preds, np.std(est_preds, axis=1)
        else:
            return preds


def _get_scalar_from_pif(
    system: System = None, name: str = None, cast: Callable = float
) -> float:
    prop = list(filter(lambda x: name.lower() in x.name.lower(), system.properties))
    if not prop:
        return
    return cast(prop[0].scalars[0].value)


def _get_1d_array_from_pif(
    system: System = None, name: str = None, cast: Callable = float
) -> np.ndarray:
    prop = list(filter(lambda x: name.lower() in x.name.lower(), system.properties))
    if not prop:
        return
    return np.array([cast(s.value) for s in prop[0].scalars])


def _get_2d_array_from_pif(
    system: System = None, name: str = None, cast: Callable = float
) -> np.ndarray:
    prop = list(filter(lambda x: name.lower() in x.name.lower(), system.properties))
    if not prop:
        return
    array = [[cast(s.value) for s in v] for v in prop[0].vectors]
    return np.array(array)


def _get_tag_value_from_pif(system: System = None, tag_key: str = None) -> str:
    tag = list(filter(lambda x: tag_key.lower() in x.lower(), system.tags))
    if not tag:
        return
    return tag[0].split(":")[-1]


def get_binding_energy(system: System) -> float:
    """Returns the binding energy of the adsorbate for the input catalyst PIF."""
    return _get_scalar_from_pif(system=system, name="Binding Energy")


def get_symbols(system: System) -> List[str]:
    """Returns the list of atomic symbols from the input catalyst PIF."""
    return _get_1d_array_from_pif(system=system, name="Symbols", cast=str)


def get_cell(system: System) -> np.ndarray:
    """Returns the cell vectors from the input catalyst PIF."""
    return _get_2d_array_from_pif(system=system, name="Cell Vectors", cast=float)


def get_positions(system: System) -> np.ndarray:
    """Returns the atomic positions from the input catalyst PIF."""
    return _get_2d_array_from_pif(system=system, name="Positions", cast=float)


def pif_to_atoms(system: System) -> ase.Atoms:
    """Converts a pif.System object into an ase.Atoms object and returns it."""
    symbols = get_symbols(system)
    cell = get_cell(system)
    positions = get_positions(system)
    atoms = ase.Atoms(symbols=symbols, cell=cell, positions=positions)
    return atoms


def get_substrate(system: System) -> str:
    """Returns the SAA substrate from the input catalyst PIF."""
    return _get_tag_value_from_pif(system=system, tag_key="substrate")


def get_dopant(system: System) -> str:
    """Returns the SAA dopant from the input catalyst PIF."""
    return _get_tag_value_from_pif(system=system, tag_key="dopant")


def db_to_design_space(
    basedir: str = ".",
    db_name: str = "dft_data.db",
    adsorbate: str = "N",
    saa_systems: List[str] = None,
) -> DesignSpace:
    """
    Takes an ASE DB containing all obtained DFT data as PIFS.
    A DesignSpace is returned containing all SAAs given by
    `saa_systems` with labels added from the ASE DB as appropriate
    """
    # default to only using 111 for fcc, 100 for bcc, 0001 for hcp
    substrate_species = [saa[0] for saa in saa_systems]
    facet_lookup = {"fcc": "fcc111", "bcc": "bcc110", "hcp": "hcp0001"}
    cs_library = {
        sp: reference_states[atomic_numbers[sp]].get("symmetry")
        for sp in substrate_species
    }
    facets = {sp: facet_lookup[cs_library[sp]] for sp in substrate_species}
    logging.debug(f"Facets: {facets}")

    # Construct full design space: structures for all saa systems
    design_space = {}
    for saa_system in saa_systems:
        saa_key = "-".join(saa_system)
        design_space[saa_key] = {}
        sub, dop = saa_system
        facet = facets[sub]
        _facet = facet.strip("fcc").strip("bcc").strip("hcp")
        saa_dict = generate_saa_structures(
            host_species=[sub],
            dopant_species=[dop],
            facets={sub: [_facet]},
            write_to_disk=False,
            default_lat_param_lib="beefvdw_fd",
        )
        design_space[saa_key]["structure"] = saa_dict[sub][dop][facet]["structure"]
        design_space[saa_key]["label"] = np.nan

    db_path = os.path.join(basedir, db_name)
    dft_data_db = ase.db.connect(db_path, type="json")

    for hit in dft_data_db.select(adsorbate=adsorbate):
        hit_pif = pif.loado(hit.data)
        saa_key = hit.saa_key

        atoms = pif_to_atoms(hit_pif)
        # remove adsorbate atoms from the structure (magpie featurization of
        # only the SAA composition to be utilized here for now)
        adsorbate_idxs = []
        for atom in atoms:
            if atom.symbol == adsorbate:
                adsorbate_idxs.append(atom.index)
        adsorbate_idxs.sort(reverse=True)
        for idx in adsorbate_idxs:
            del atoms[idx]

        # model on adsorbate binding energies
        energy = get_binding_energy(hit_pif)
        print(saa_key, atoms, energy)

        design_space[saa_key].update({"structure": atoms, "label": energy})

    # Convert design space dict into lists with 1:1 correspondence (sorting is
    # important here!)
    structures = [v["structure"] for _, v in sorted(design_space.items())]
    labels = np.array([v["label"] for _, v in sorted(design_space.items())])

    return DesignSpace(design_space_structures=structures, design_space_labels=labels)


def update_saa_candidates(
    candidate_structures: List[ase.Atoms] = None, basedir: str = "."
):
    """
    Updates the `saa_candidate_systems.json` with the latest candidate
    system selected for evaluation
    """
    saa_candidates_json = os.path.join(basedir, "saa_candidate_systems.json")
    if os.path.isfile(saa_candidates_json):
        with open(saa_candidates_json, "r") as fr:
            saa_systems = json.load(fr)
        saa_systems = set(saa_systems)
    else:
        saa_systems = set()

    for atoms in candidate_structures:
        formula_dict = dict(Counter(atoms.get_chemical_symbols()))
        assert len(formula_dict) == 2
        saa_system = "-".join(
            [x[0] for x in sorted(formula_dict.items(), key=lambda x: -1 * x[1])]
        )
        print(f"New SAA candidate system: {saa_system}")
        saa_systems.add(saa_system)

    saa_systems = sorted(list(saa_systems))
    with open(saa_candidates_json, "w") as fw:
        json.dump(saa_systems, fw, indent=2)


def run_sequential(
    adsorbate: str = "N",
    basedir: str = ".",
    db_name: str = "dft_data.db",
    force_iteration: bool = False,
):
    """Runs one iteration of sequential learning."""
    saa_systems = list(it.permutations(ELEMENTS, r=2))
    saa_systems = [saa for saa in saa_systems if saa[0] != "Ti"]
    design_space = db_to_design_space(
        basedir=basedir,
        db_name=db_name,
        adsorbate=adsorbate,
        saa_systems=saa_systems,
    )
    design_space.write_json_to_disk(write_location=basedir)
    existing_acsl_file = os.path.join(basedir, "acsl.json")
    if os.path.isfile(existing_acsl_file):
        learner = SequentialLearner.from_json(existing_acsl_file)
        first_iteration = False
    else:
        featurizer = Featurizer(featurizer_class=ElementProperty, preset="magpie")
        regressor = RandomForestRegressor()
        predictor = Predictor(regressor=regressor, featurizer=featurizer)
        candidate_selector = CandidateSelector(
            acquisition_function="MLI",
            target_window=(-1.1767, -0.5767),
            include_hhi=True,
            hhi_type="reserves",
            include_segregation_energies=True,
        )
        learner = SequentialLearner(
            design_space=design_space,
            predictor=predictor,
            candidate_selector=candidate_selector,
        )
        first_iteration = True
    if (
        not np.array_equal(
            learner.design_space.design_space_labels,
            design_space.design_space_labels,
            equal_nan=True,
        )
        or first_iteration
        or force_iteration
    ):
        learner.design_space = design_space
        learner.iterate()
        candidate_structures = learner.candidate_structures
        update_saa_candidates(
            candidate_structures=candidate_structures, basedir=basedir
        )
    learner.write_json_to_disk(write_location=basedir, json_name="acsl.json")

