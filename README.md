# CityFlow-NL: City-Scale Retrieval of Vehicles by Natural Language Descriptions

This dataset and codes are curated for the CVPR 2022 Submission 775.

## Contents in this repository
`data/extract_vdo_frms.py` is a Python script that is used to extract frames
from videos provided in [1]. Please use this script to extract frames, so that
the path configurations in JSON files are consistent.

`data/train-tracks.json` is a dictionary of all 2,498 vehicle tracks in the
training split. Each vehicle track is annotated with three natural language
descriptions of the target and is assigned a universally unique identifier
(UUID).  The file is structured as
```json
{
  "track-uuid-1": {
    "frames": ["file-1.jpg", ..., "file-n.jpg"],
    "boxes": [[742, 313, 171, 129], ..., [709, 304, 168, 125]],
    "nl": [
      "A gray pick-up truck goes ...", 
      "A dark pick-up runs ...", 
      "Pick-up truck that goes ..."
    ]
  },
  "track-uuid-2": ...
}
```
The files under the `frames` attribute are paths in the CityFlow Benchmark [1].


`data/test-tracks.json` contains 530 tracks of target vehicles. The structure
of this file is identical to the training split, except that the natural
language descriptions are removed.

`data/test-queries.json` contains 530 queries. Each consists of three natural
language descriptions of the vehicle target annotated by different annotators.
Each query is assigned a UUID that is later used in results submission.  The
structure of this file is as follows:
```json
{
  "query-uuid-1": [
    "A dark red SUV drives straight through an intersection.",
    "A red MPV crosses an empty intersection.",
    "A red SUV going straight down the street."
  ],
  "query-uuid-2": ...
}
```

The `retrieval/baseline` directory contains the baseline model that measures
the similarity between language descriptions and frame crops in a track.
Details of this model can be found in Section 4.3.

The `retrieval/baseline_nk2` directory contains the quad-stream model that
measures the similarity between language descriptions and frame crops, optical
flow, relations in a track. Details of this model can be found in Section 4.5.

Main python scripts for running the baseline and the quad-stream model are
called `workflow.py`.

```bash
python retrieval/baseline/workflow.py --config_file=experiments/baseline-retrieval-train.yaml

python retrieval/baseline_mk2/workflow.py --config_file=experiments/mk2/sc_train.yaml
```


## References
[1] Tang, Zheng, et al. "CityFlow: A city-scale benchmark for multi-target
multi-camera vehicle tracking and re-identification." CVPR. 2019.
