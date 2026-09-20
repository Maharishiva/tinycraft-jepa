# ICLR 2027 operational rules, verified 19 September 2026

**Submission eligibility is time-sensitive.** The official abstract deadline was **18 September 2026, 23:59 AoE**, equivalent to **19 September 2026, 11:59 UTC / 13:59 Belgrade**. The live clock was 19 September 19:43 UTC during this check, so that deadline has passed. An actual ICLR 2027 submission therefore needs an already-submitted genuine abstract. This does not prevent preparing the manuscript. Full papers and supplements are due **25 September 2026, 23:59 AoE** (**26 September, 11:59 UTC / 13:59 Belgrade**). Reviews: 5 November; author discussion: 5–18 November; decisions: 16 December. [Official call](https://iclr.cc/Conferences/2027/CallForPapers).

## Submission checklist

- Initial main text: **at most 9 pages**. Rebuttal and camera-ready: 10 pages. References and appendices have no page limit; appendices follow references and reviewers need not read them. The guide contains stale FAQ language calling 10 pages the initial limit; the explicit formatting rule and 2027 template agree on 9, which should govern.
- Keep the paper and all supplements anonymous. Refer to related work by the same authors in the third person. Anonymous code archives/repositories are allowed. Demo links must be anonymous and must not track visitors.
- Author membership froze at the abstract deadline. Author order, title, and abstract can change until the paper deadline, but the final paper must remain substantively consistent with the registered abstract.
- Include the required AI-use statement. Ethics and reproducibility statements are recommended. These sections do not count toward the main-text limit.

[Official author guidelines](https://iclr.cc/Conferences/2027/AuthorGuidelines).

## Formatting and visual design checklist

- Use the unmodified **2027** LaTeX style, not a visually imitated HTML export. The downloaded official source is at [official style copy](formatting/iclr2027/).
- Preserve US Letter pages, 5.5 × 9 inch text area, 10-point body text with 11-point spacing, and page numbers. Do not alter margins, text dimensions, or font sizes to fit content. Keep `\iclrfinalcopy` commented out for submission.
- Figures may use color. Design redundant encodings (labels, markers, line patterns) so captions and claims remain intelligible in grayscale. Captions belong **below figures**. Tables must be centered and legible, with captions **above tables**. Use sentence case and consecutive numbering.
- The template does not explicitly prohibit colored table cells. Conservative light tints, dark text, and explicit values are compatible with its legibility requirements; this is a design interpretation, not an explicit table-color authorization. Keep color styling inside figure/table assets rather than changing conference typography or page backgrounds.
- The required AI statement should be at most one page.

[Official style archive](https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip), inspected `iclr2027_conference.tex` lines 91–131, 179–211, 352–362, 397–412 and the `.sty` page dimensions.

## AI disclosure for this project

Disclosure is required both in the manuscript and submission form. Required categories include research-framework development, hypothesis refinement, experimental/methodological design or feedback, method implementation, data processing, and results interpretation; mathematical work and translation are also required when applicable. Recommended disclosures include literature search/synthesis, figures, drafting/editing, code assistance, structure, and references. This project should disclose substantive research assistance, not describe it merely as copyediting. Authors retain responsibility for correctness, originality, and faithful representation. Do not claim human verification that has not happened.

Suggested factual starting point, to update against the completed work: “We used generative AI tools to assist literature review, refine experimental hypotheses, design and implement evaluation procedures, interpret results, draft and edit the manuscript, and produce scientific visualizations. The final submission records the verification performed by the authors and their responsibility for its contents.” Replace the last sentence with concrete completed checks before submission; agent review alone is not author verification.

[Official AI policy for authors](https://iclr.cc/Conferences/2027/AIPolicyForAuthors).

## Delivery recommendation

Create two renderings from one audited numerical source: a polished HTML reading/review copy and a conference-compliant LaTeX PDF. HTML is a useful visual-review artifact; it does not substitute for the mandated submission format. Verify PDF page count, caption placement, grayscale legibility, anonymity, and every reported number against retained experiment artifacts. Independent agent assessments are useful critical feedback; they cannot certify an ICLR acceptance score.


Standalone archive: official PDFs, extracted text, inspected renders and source hashes are retained in [reading/](reading/) and [reading/manifest.json](reading/manifest.json). Original `/tmp` locations above record the reading process; the retained archive is the deliverable.
