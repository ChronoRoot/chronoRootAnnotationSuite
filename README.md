# ChronoRoot Annotation Suite

An open-source desktop tool for correcting automatic root segmentations on petri-dish scanner images, and optionally measuring and comparing root traits across experiments.

**Website:** [chronoroot.github.io](https://chronoroot.github.io/) — full user guide and documentation.

## Annotation workflow

1. **Annotation** — fix plants, shapes, and root-part labels.
2. **Centerline Cleanup** — compare panels; apply if refinement looks better.
3. **Root Tracing Check** — confirm each plant traces correctly.
4. **Finish Annotation** — done.

## Traits workflow

1. **Plant Metadata** — plate info, image scale, genotypes.
2. **Measure** then **Export**.
3. **Phenomics Inspector** — review per plant.
4. **Batch Reports** — compare plates.

Measure requires the plate to be finished first.

## Launch

```bash
./main.sh
```

Or: `python main.py`

Config: `~/.config/chronoRootAnnotationSuite/config.json`

## Input and output files

See [Tutorial/tutorial.md](Tutorial/tutorial.md) for file formats (scanner image, model segmentation, saved annotations, and analysis exports).

## License and updates

See [chronoroot.github.io](https://chronoroot.github.io/). In-app **About → Check for Updates** runs `git pull` on git checkouts.
