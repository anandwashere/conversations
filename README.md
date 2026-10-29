# Can AI Traverse the "Right" Graph of Research Conversations?

Prototype accompanying a proposal to the *AI-Enabled Frontiers in Organization Science* special issue of *Organization Science*. Authors are anonymized for review.

**Claim under test.** Retrieval by semantic similarity, the basis of LLM literature tools, misidentifies research conversations. Conversations are constituted socially (who works with, for and alongside whom), not only lexically or semantically.

**Approach.** Within a keyword frame (e.g. papers whose title or abstract contains *technolog\_*), we draw paper pairs under different retrieval conditions. A domain expert, blind to condition, judges whether each pair belongs to the same research conversation. The outcome is each condition's **precision**: the share of pairs judged to be in the same conversation.

## Four layers of referencing

| layer | what links two papers | operationalized as |
|---|---|---|
| lexical | shared words | `L0`: random pairs from the keyword frame (database keyword search) |
| semantic | shared meaning in text | `T1`: same BERTopic topic over sentence embeddings (the LLM view) |
| social | shared community | `T1C1`: same topic *and* same coauthor community (sociosemantic search) |
| pragmatic | standpoint, timing, the politics of citation | the expert judgments themselves; in the full design, disagreement among coders is treated as evidence of this layer |

Each step up tests whether a layer adds information the one below it missed.

## Designs

The notebook runs either design, selected by `DESIGN` in the configuration cell.

| | Prototype (`DESIGN = "prototype"`) | Full design (`DESIGN = "full"`, Stage 2) |
|---|---|---|
| cells | `L0`, `T1`, `T1C1`, sampled directly | `L0` plus the topic × community 2×2 (`T0C0`, `T1C0`, `T0C1`, `T1C1`); semantic (`T1`) and social (`C1`) retrieval derived as mixtures |
| pairs | 10 per cell, 30 per track | ~30 per cell |
| tracks | `knowledge` (*knowledg\_*), `technology` (*technolog\_*) | more keyword tracks |
| coders | one expert per track | 10–20 tenure-track faculty, several per track; agreement measured and related to coders' positions in the field |
| corpus | 20 journals, 2000–2023 (below) | more journals and adjacent fields, a longer span, plus citation, conference-attendance and academic-lineage data for selected areas |
| key contrasts | `T1` vs `T1C1` (does social structure raise precision?); `L0` vs `T1` | `T1C0` vs `T1C1` (social signal, text held constant); `T1` vs `C1`; `L0` vs `T1`; `T1` matched to `T1C1` on cosine similarity |

The prototype shows direction and effect size, with intervals, plus qualitative cases. It is not a confirmatory test.

**Safeguards**
- No pair shares an author.
- No paper appears in more than one pair, across both tracks.
- Labelers are blind to condition: pairs are shuffled, and the cell key is kept in a separate file.
- Author names are hidden from labelers by default (`SHOW_AUTHORS = False`).
- Pairs are drawn uniformly from each cell's eligible pairs.
- The topic model and coauthor network are fitted on the full corpus, not on the keyword frames.

## Prototype data and current configuration

**Corpus:** 28,620 papers (27,955 with abstracts), 35,476 authors, 2000–2023, from 20 journals:
- *Academy of Management Journal*, *Academy of Management Review*, *Administrative Science Quarterly*
- *Information and Organization*, *Information Systems Research*, *Journal of Applied Psychology*
- *Journal of International Business Studies*, *Journal of Management*, *Journal of Management Inquiry*
- *Journal of Management Studies*, *Journal of Organizational Behavior*, *Journal of Strategic Information Systems*
- *MIS Quarterly*, *Organizational Behavior and Human Decision Processes*, *Organization Science*
- *Organization Studies*, *Research Policy*, *Strategic Management Journal*, *Strategic Organization*
- *Technological Forecasting and Social Change*

| component | setting | result |
|---|---|---|
| embeddings | `all-mpnet-base-v2`, title + abstract | 768-d, L2-normalized |
| topics | BERTopic; HDBSCAN `min_cluster_size=20`, `min_samples=5` | 290 topics; 35.7% outliers |
| communities | Leiden (RB-configuration), resolution 1.0, on the full coauthor network | 2,925 non-singleton communities |
| frames | *Technological Forecasting and Social Change* excluded from the frames (still in the topic model and network) | `knowledge`: 4,336 papers; `technology`: 3,578 papers |
| topic rule | `"all"`: any non-outlier topic can feed `T1` | `"top_words"` kept as a robustness check |

These figures are from the current run. Change the configuration and they change too; the run config is saved to `outputs/run_config_<design>.json`.

## Layout
```
data/
  papers.parquet         28,620 papers (Scopus export): title, abstract (cleaned), abstract_raw, journal, year, cited_by
  authorships.parquet    79,470 paper–author rows keyed on Scopus author ID
notebooks/
  Conversations_AIEFOS_Proposal_Prototype.ipynb
outputs/                 pair sample, blind labeling sheets, cell keys, cases, figures, run config
labels/                  filled labeling sheets: <track>_<labeler>.xlsx
cache/                   embeddings + topic assignments per topic setting (gitignored)
```

## Run
```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
jupyter lab notebooks/Conversations_AIEFOS_Proposal_Prototype.ipynb
```
- On Apple silicon, embeddings run on MPS.
- The first run embeds the corpus and fits the topic model, taking a few minutes each; later runs load both from `cache/`.
- Topic runs are cached per (`min_cluster_size`, `min_samples`) setting, so earlier granularities stay available for the robustness check in section 9.

## Labeling workflow
1. Run sections 1–7. The blind sheets are written to `outputs/<track>_labeling_<labeler>.xlsx`.
2. The coder fills in `label` (S = same conversation, D = different, U = can't tell), plus `confidence` (1–3) and `notes`. Notes should name the conversation(s) involved.
3. Save the filled sheet as `labels/<track>_<labeler>.xlsx` and rerun section 8.

A second coder on the same track saves `labels/<track>_<coder>.xlsx`; the notebook then reports Cohen's κ. Until real labels exist, section 8 runs on simulated labels, and every output is stamped SIMULATED. Section 8 also writes cases for the memo to `outputs/cases_<track>.csv`. These are semantic false positives (high-similarity, same-topic pairs judged different) and sociosemantic hits (`T1C1` pairs judged the same).
