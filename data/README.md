# Data

Run `../scripts/download_gist1m.sh` to fetch GIST1M (TEXMEX corpus, Jégou et al.) from
`ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz` (2,740,172,684 bytes). The archive
is kept in `gist1m_download/` and the four `.fvecs`/`.ivecs` files are extracted into
`gist1m/`. Paths are configurable in `config/default.yaml` (`dataset.dir`, `dataset.files`).

Validate with `python -m src.datasets` (checks every record header, sizes, finiteness,
ground-truth shape/range/duplicates, and brute-force agreement on 20 random queries).
