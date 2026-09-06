# Ryan's writing style (reference for every manuscript edit)

Derived from `multiscale_nature_comm/sn-article-natcomsci.tex`, `neurovlm_nature_comm/sn-article.tex`,
and the Introduction of `overleaf/main.tex`. Match this; do not "improve" on it.

## Structure

- Section and subsection titles are short noun phrases in Title Case, one to three words:
  `Segmentation \& Parcellation`, `Morphometric Features`, `Clinical Application`, `Broader Impact`,
  `Model Architecture`, `Contrastive Learning`, `Datasets`, `Training`, `Limitations`, `Future`.
  Never a sentence, never a claim, never lower-case prose.
- Figure captions open with a short bold title, a noun phrase ending in a period, then bold panel letters:
  `\caption{\textbf{Model architecture.} \textbf{a)} NeuroVLM maps ...}` or `\textbf{a) PubMed.} Contrastive recall curves ...`.
  The title names the figure's content; it does not state a finding.
- Paragraph shape: a topic sentence that states the fact or result, supporting sentences with the numbers,
  and often a closing sentence that says what the result shows
  ("These results show that ...", "This represents ...", "That is, ...").
- Results report what was done and found, in past tense, with the figure reference in parentheses at the end of
  the sentence: "ScaleSurfer achieved the highest agreement with FreeSurfer for cortical parcellation (Dice$=0.81$)".
- Methods are procedural and exact: "X was computed as", "We used", "was evaluated using", settings listed,
  "Exact ... are reported in \tabpanel{...}".
- Discussion opens with the contribution ("In this work, we introduce ...", "We demonstrate that ...",
  "We show ...", "Finally, we show ..."), then implications, then `Limitations` ("One major limitation is
  that ...", "There is also ...", "Finally, ...") and `Future`.

## Sentences

- Plain declaratives of medium length. One idea per sentence. Explicit connectors: "However,", "In addition,",
  "Finally,", "Notably,", "For example,", "In contrast,", "Together,", "That is,".
- Concrete subjects: the model, the estimate, the recording, the timescale, "we". Not abstractions
  ("the obstacle", "the difficulty", "the interpretation").
- Terms are defined before they are used, then used consistently. No undefined evaluative words
  ("natural", "honest", "the point").
- Qualitative words are conventional and mild: high, strong, moderate, close, modest.
- Second person and analogies are fine in the Introduction ("you can't predict future values", the temperature
  example). Occasional plain enthusiasm is fine ("incredibly large scale"), but not cleverness.

## What Ryan does not write

- Aphorisms, chiasmus, slogans, or contrasts built for effect
  ("Reproducibility is not recoverability"; "the spectrum is where it can be avoided"; "Importance, not pole type, decides").
- Teaser or fragment sentences that set up the next one ("A short window does more than add noise.").
- "not X but Y" and "X is one thing, Y is another" constructions used rhetorically.
- Anthropomorphized methods ("the fit reports that ...", "the estimator chooses").
- Abstract-noun openers ("The obstacle is removable, because ..."; "The reason is structural.").
- Parallel two-sentence pairs for rhythm ("Simulations show what can go wrong. Recordings show how often it does.").
- Opinion stated as fact ("the coefficients are not the natural description of the model").
- Section titles that are sentences; caption titles that are findings.
- LLM filler: "It is important to note", "This highlights", "These findings underscore", "Taken together".
- Rhetorical questions.

## Quick test before saving a paragraph

1. Does the first sentence state a fact or result, with a concrete subject?
2. Could every sentence be read aloud to a colleague without sounding like a maxim?
3. Are the numbers in the sentence, with units, and the figure reference at the end?
4. Is every technical term one that was defined earlier in the paper?
5. Is the title a noun phrase of one to three words in Title Case?
