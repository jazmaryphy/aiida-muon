# Commands...

## - Run structural relaxations

run:

`python muon.py --structure "structure" --title="title" --group="group" --sc-size=sc_size --sc-gen-mode grid:grid_size --charged "bool" --upf-family pseudo_family --pw-code="pw_codename" --runtime-options runtime_options --task sites sites.yaml` 
  
## - Run hyperfine calculation

run:

`python muon.py --structure "structure" --group="group" --sc-size=sc_size --if_equivalent_sites "bool" --if_with_distortions "bool"  --pp-code="pp_codename --projwfc-code="projwfc_codename"  --runtime-options runtime_options --task 'hyperfine->UUID' hyperfine.yaml`

where `UUID` can be of QE relax `PwCalculation` that of `group` define above. The `UUID` of a `group` will perform hyperfine calculations for all relax structure stored in `group`.

## - Query and generate cluster based on symmetry.

run:

`python muon.py --structure "structure" --sc-size=sc_size --energy_threshold=e_prec --symprec=mu_prec --if_with_energy "bool" --task 'query|cluster->UUID'  cluster.yaml`

where `UUID` is that of a `group`.

## - Query results of relax/hyperfine calculations and their corresponding states.

run:

`python muon.py --structure "structure" --workchain_name=name --process_status=status --task 'query->UUID'  query.yaml`

where `UUID` is that of a `group`


## - Calculate field

run:

`python muon.py --structure "structure" --sc-size=sc_size --if_equivalent_sites "bool" --if_scale_moment "bool"  --if_scale_contact "bool" --if_with_contact "bool" --if_with_distortions "bool" --file_name="file" --task 'fields->UUID'  field.yaml`

where `UUID` can be that of `relax`, `hyperfine` or `group` node. If `group` is given will calculate hyperfine for all structures.


## where:

  1. `"structure"`: Is the structure object e.g `"structure"=Fe.mcif` for magnetic structure.
  2. `"title"`: Name of title of calculation.
  3. `group`: Name of the group that will be used to store AiiDA UUID of calculation. This ensure we organize all our data that can be track easily. e.g `group=Fe`. For more check [this tutorials](https://aiida-tutorials.readthedocs.io/en/tutorial-2021-intro/sections/getting_started/basics.html#fundamentals-basics).
  4. `sc_size`: Supercell size in `a`, `b` and `c` crystallographic direction e.g. `sc_size="3 3 3"` to perform structural relaxations.
  5. `grid_size`: An integer value of grid size to sample initial muon (H) positions using uniform sampling in a function `aiida_muesr_ugrid.py`.
  6. `charged`: If `True` to initial charge state of muon (Hydrogen) in our DFT calculation.
  7. `pseudo_family`: Assign the name of pseudopotential data saved on AiiDA database. For more info. check [this tutorials](https://aiida-tutorials.readthedocs.io/en/tutorial-2021-intro/sections/getting_started/basics.html#fundamentals-basics).
  8. `pw_codename`: The name of QE `PW` code to perform structural relaxations. For more check this [this tutorials](https://aiida-tutorials.readthedocs.io/en/tutorial-2021-intro/sections/getting_started/basics.html#fundamentals-basics).
  9. `pp_codename`: The name of QE `PP` postprocessing.
  9. `projwfc_codename`: The name of QE `PROJWFC` postprocessing
  9. `runtime_options`: Computing hours required e.g `runtime_options="1 259000 | -nk 1"` where `"1 259000 | -nk 1" == "number_of_nodes time_in_seconds | -nk number_of_pools"`.
  3. `e_prec`: Energy difference threshold in `eV`, e.g. `e_prec=0.001`
  4. `mu_prec` Symmetry threshold to check distance, e.g. `mu_prec=0.01`
  1. `name`: Is the name calculation to be query, e.g. `name = "relax"` or `"contact"`
  2. `status`: Is the status of each calculation, e.g. `status="all"`, `"finished"` or `"excepted"`.  
  0. `if_with_energy`: If `True` to query energy, e.g. `"bool" = True` 
  1. `if_equivalent_sites`: If `True` to generate symmetrized structures, e.g. `"bool" = True` 
  2. `if_scale_moment`: If `True` to scale calculated `DFT` moments e.g. `"bool" = True` 
  3. `if_with_contact`: If `True` to use contact hyperfine fields, e.g. `"bool" = True` 
  4. `if_scale_contact`: If `True` to scale contact hyperfine fields, e.g. `"bool" = True` 
  5. `if_with_distortions`: If `True` to consider distortions, e.g.  `"bool" = True` 
  6. `"file"`: The name of material to collect experimental data. This name most coincide with the data store in `observed_muon_precession.py` function. 
  7. `UUID`: The assigned primary key (pk) or universally unique identifier (uuid) that identifies the node of a `group`, `hyperfine` or `relax` calculations store in AiiDA database. For more check [this tutorials](https://aiida-tutorials.readthedocs.io/en/tutorial-2021-intro/sections/getting_started/basics.html#fundamentals-basics).
