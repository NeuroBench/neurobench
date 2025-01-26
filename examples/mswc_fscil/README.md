Multilingual Spoken Word Corpus (MSWC) Few-Shot Class-Incremental Learning (FSCIL) Benchmark
=======================================================================
The MSWC FSCIL task tests a model's ability to learn a base dictionary of keywords, then successively learn new keywords using few training samples while retaining memory of all learned words.
More information is available in the MSWC_tutorial notebook and associated script `mswc_fscil.py`.
The task script and infrastructure were built by Maxime Fabre and Douwe den Blanken.

## Version Notice
The MSWC NeuroBench dataloader uses torchaudio to load audio files. Based on the the backend that torchaudio uses to decompress the data files, waveforms can appear differently. The SNN is particularly sensitive to this. The SNN was trained and evaluated using `torchaudio=2.0.2` and the `sox_io` backend for loading.