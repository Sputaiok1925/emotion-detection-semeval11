# Translation review

All 8,896 cached source/translation pairs were screened with inexpensive checks for empty output, suspicious length ratios, lost negation cues and repeated characters. Candidate pairs were compared against their originals. A reproducible random sample of 25 pairs per language (Python random.Random(42), cache insertion order) was also inspected. These heuristics are not proof of translation accuracy. Not all pairs were semantically reviewed.

139 unique translations were corrected with AI assistance to restore omitted content or repair clear meaning errors. No emotion labels were consulted to choose a translation, and none were changed. No translation model was rerun. Original cache files are backed up here; corrections.csv records original text, old translation and replacement. This is not human-certified gold translation. Some remaining texts may contain mistranslations, slang ambiguities or lost emojis. Document AI assistance in the course project's AI appendix.

The extended dataset and its existing reports were regenerated using the project's preparation functions and cached original data. Final train: 11,640; test: 2,761. Empty texts, invalid labels, repeated full texts and exact train/test overlaps were checked. The two extra training rows compared with the previous output result from corrections distinguishing translations that previously collapsed to the same text. Original English files in data/processed were not edited. No dev split was added.

Reproducibility: retain the corrected translation cache, or apply corrections.csv by (language, original_text) to a freshly generated cache before rerunning preparation. Preserve the backups for comparison.
