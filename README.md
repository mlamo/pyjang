# Joint Analysis of Neutrinos and Gravitational waves

![logo](https://github.com/mlamo/jang/blob/main/doc/logo.png?raw=true)

[![tests](https://github.com/mlamo/jang/actions/workflows/tests.yml/badge.svg)](https://github.com/mlamo/jang/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/mlamo/jang/branch/main/graph/badge.svg?token=PVBSZ9P7TR)](https://codecov.io/gh/mlamo/jang)

# Installation

* Clone the repository: ``git clone https://github.com/mlamo/jang.git``
* Install the package: ``cd jang && pip install -e .``

# Step-by-step usage

## Parameters

* Create/use a YAML file with all needed parameters (example: ``examples/input_files/config.yaml``)
* Load the parameters:
```python
from jang.parameters import Parameters
pars = Parameters("examples/parameter_files/path_to_yaml_file")
```

* Select the neutrino spectrum and jet model:
```python
pars.set_models("x**-2", jang.conversions.JetIsotropic())
```
(the list of available jet models is available in ``jang/conversions.py``)

## Detector information
   
* Create/use a YAML file with all relevant information (examples in ``examples/input_files/DETECTORNAME/detector.yaml``)
* Create a new detector object:
```python
from jang.neutrinos import Detector
det = Detector(path_to_yaml_file)
```

* Acceptances have to be defined for the spectrum to consider:
   * if they already exist in npy or ndarray format (one for each sample), they can directly be loaded:
   ```python
   det.set_acceptances(npy_path_or_ndarray, spectrum="x**-2")
   ```

   * otherwise, it can be estimated using an object of a class derived from EffectiveAreaBase, as illustrated for Super-Kamiokande in ``examples/superkamiokande.py``

* Any observation can be set with the following commands, where the two arguments are arrays with one entry per sample:
```python
# different background models are available: BackgroundFixed(b0), BackgroundGaussian(b0, deltab), BackgroundPoisson(Noff, Nregionsoff)
bkg = [BackgroundFixed(0.51), BackgroundFixed(0.12)]
det.set_observations(n_observed=[0,0], background=bkg)
```

## GW information

* GW database can be easily imported using an existing csv file (see e.g., ``examples/input_files/gw_catalogs/database_example.csv``):
```python
import jang.gw
database_gw = jang.gw.Database(path_to_csv)
```

* An event can be extracted from it:
```python
gw = database_gw.find_gw(name_of_gw, pars)
```

## Compute limits

* Limit on the incoming neutrino flux (where the last optional argument is the local path -without extension- where the posterior could be saved in npy format):
```python
jang.limits.get_limit_flux(det, gw, pars, path_to_file)
```

* Same for the total energy emitted in neutrinos:
```python
jang.limits.get_limit_etot(det, gw, pars, path_to_file)
```

* Same for the ratio fnu=E(tot)/E(rad,GW):
```python
jang.limits.get_limit_fnu(det, gw, pars, path_to_file)
```

## Results database
   
* Create/open the database:
``` python
import jang.results
database_res = jang.results.Database(path_to_csv)
```

* Add new entries in the database:
```python
database_res.add_entry(det, gw, pars, limit_flux, limit_etot, limit_fnu, path_to_flux, path_to_etot, path_to_fnu)
```

* Save the database:
```python
database_res.save()
```

# Full examples

Some full examples are available in `examples/`:
* `superkamiokande.py` provides a full example using Super-Kamiokande public effective areas from [Zenodo](https://zenodo.org/records/4724823).
* `full_example.ipynb` provides a step-by-step example to get Super-Kamiokande/ANTARES sensitivities and perform a combination. The ANTARES acceptance are rough estimates from [JCAP 04 (2023) 004](https://arxiv.org//abs/2302.07723).