# Vasculature API
 
[![pipeline status](https://bbpgitlab.epfl.ch/molsys/vascpy/badges/main/pipeline.svg)](https://bbpgitlab.epfl.ch/molsys/vascpy/-/commits/main)


[![coverage report](https://bbpgitlab.epfl.ch/molsys/vascpy/badges/main/coverage.svg)](https://bbpgitlab.epfl.ch/molsys/vascpy/-/commits/main) 

Introduction
------------

vascpy is a library for reading and writing vasculature datasets using two alternative representations: section-centered and edge-centered. It supports the following respective formats:

* H5 Morphology (see [specification](https://bbpteam.epfl.ch/documentation/projects/Morphology%20Documentation/latest/h5vasculature.html))
* SONATA node population of edges (see [specification](https://bbpteam.epfl.ch/documentation/projects/Circuit%20Documentation/latest/sonata_tech.html))

The vascpy provides two classes: `PointVasculature` and `SectionVasculature` that allow for reading and writing edge-centered and section-centered datasets respectively, as well as converting between them.

Basic usage
-----

Load and write an h5 morphology file:

```python
from vascpy import SectionVasculature

v = SectionVasculature.load_hdf5("sample.h5")

print(v.points)
print(v.diameters)
print(v.connectivity)
print(v.sections)

v.save_hdf5("sample2.h5")
```

Load and write an h5 SONATA file:
```python
from vascpy import PointVasculature

v = PointVasculature.load_sonata("sample_sonata.h5")

print(v.node_properties)
print(v.edge_properties)
print(v.points)
print(v.edges)
print(v.edge_types)
print(v.segment_points)
print(v.segment_diameters)
print(v.area)
print(v.volume)

v.save_sonata("sample_sonata2.h5")
```

Representation conversions
-----------

vascpy allows the conversion between the two representations:

```python
from vascpy import PointVasculature
point_vasculature = PointVasculature.load_hdf5("sample_sonata.h5")

section_vasculature = point_vasculature.as_section_graph()
point_vasculature = section_vasculature.as_point_graph()
```

Create and save an edge-centered vascular graph
-----------------------------------------------

```python
import numpy as np
import pandas as pd
from vascpy import PointVasculature

node_properties = DataFrame({
    'x': np.array([0., 1., 2.]),
    'y': np.array([3., 4., 5.]),
    'z': np.array([6., 7., 8.]),
    'diameter': np.array([0.1, 0.2, 0.3])
})

edge_properties = pd.DataFrame({
    'start_node': np.array([0, 0, 1]),
    'end_node': np.array([1, 2, 2]),
    'type': np.array([0, 1, 1])
})

v = PointVasculature(node_properties=node_properties, edge_properties=edge_properties)
v.save_sonata('my_vasculature.h5')
```
