from dingo.model import Model
from dingo.model.llm.text_quality.base_text_quality_v2 import BaseTextQualityV2


@Model.llm_register("LLMTextQualityV8")
class LLMTextQualityV8(BaseTextQualityV2):
    """Multi-label text quality evaluator with a self-contained prompt."""

    _metric_info = {
        "category": "Pretrain Text Quality Assessment Metrics",
        "metric_name": "LLMTextQualityV8",
        "description": "Multi-label impact-driven text quality evaluation for LLM pretraining",
        "paper_title": "WanJuanSiLu: A High-Quality Open-Source Webtext Dataset for Low-Resource Languages",
        "paper_url": "https://arxiv.org/abs/2501.14506",
        "paper_authors": "Yu et al., 2025",
        "examples": "examples/llm_and_rule/llm_local.py",
        "evaluation_results": "docs/eval/prompt/redpajama_data_evaluated_by_prompt.md",
    }
    prompt = r"""
# Role
You are an expert in assessing pretraining data quality for large language models.

# Goal
Evaluate whether this text is suitable for LLM pretraining. Flag only clear, material defects that would teach incorrect structural or linguistic patterns. Do not reject text for minor imperfections, unfamiliar languages, stylistic preferences, or defects that are only speculative.

# Core Decision Policy
1. Judge only from evidence present in the input. Do not reconstruct or assume unavailable source content.
2. Identify the text type before judging it: prose, list, table, code, mathematical content, navigation, metadata, or mixed content.
3. Apply a label only when its defining defect is explicit and materially affects a non-trivial portion of the sample.
4. Respect the exclusions under each label. A low-quality topic or awkward writing is not automatically a formatting defect.
5. Return every clearly supported label. Each label must describe a distinct material defect; do not emit duplicates or speculative secondary labels. If no label is clearly supported, return only Good.

# Quality Dimensions

## 1. Completeness (完整性)
**Impact**: Broken structures prevent models from learning correct formatting patterns.

**Check for**:
- **Formula quality labels**: Evaluate extraction or representation defects in mathematical formulas, equations, chemical formulas, chemical structures, and formula-like scientific notation. Do not evaluate whether a mathematically well-formed statement is factually true.

  **Shared formula policy**:
  - Use only evidence visible in the input. Do not reconstruct a source formula from outside knowledge.
  - A defect must be explicit and materially affect understanding, parsing, or display. Do not flag a harmless isolated style difference.
  - For one damaged span, return the single label that most directly describes the defect. Do not describe the same missing brace, symbol, or fragment with multiple labels.
  - Multiple formula labels are allowed only for independent defects, such as one entirely missing formula and a different formula contaminated by a page header.
  - Ordinary prose typos, game notation, navigation, numbering, and non-formula text are outside these labels.

  **Formula_Missing** — An entire formula, equation, matrix, or chemical structure that should visibly be present is absent.
  - Flag when an introduction such as "the formula is", "we obtain", "the matrix is", or "the structure is" is followed by no formula.
  - Flag when only a formula number or placeholder such as `[Formula 1]`, `[Chemical Formula 1]`, or `(I)` remains, while following text defines variables from the absent formula.
  - Flag when a formula position is replaced by an unrelated page header, page number, or paragraph, leaving the surrounding derivation broken.
  - Flag repeated references to absent equations only when the current sample visibly depends on them and the missing content breaks the derivation.
  - Do not flag an external citation, an explicitly omitted derivation, a harmless gap in equation numbering, or a passage that merely mentions a result without promising to display it.
  - Do not infer that a formula was missing merely because a formula would be useful.

  **Formula_Partial_Loss** — A formula remains, but a meaningful mathematical or scientific component has been removed.
  - Components include variables, constants, operators, relation or mapping symbols, operands, numerator or denominator, equation sides, function arguments, sum/integral limits, piecewise branches, chemical subscripts, charges, or complete subexpressions.
  - Examples include `T(s,a,s') [0,1]` with a visibly missing mapping arrow, `x+y=` with the right-hand side absent, a fraction with no denominator, or a chemical formula with a clearly missing element count.
  - Use this label when meaningful formula content is absent. If erroneous or garbled characters remain in its place, use Formula_Token_Corruption instead.
  - If only a LaTeX delimiter, required brace, or environment terminator is missing and parsing fails, use Formula_Unparseable instead.
  - Do not flag valid shorthand, an omitted derivation, or a term that is only suspected to be absent from domain knowledge.

  **Formula_Token_Corruption** — Formula characters, numbers, variables, or symbols are misrecognized, substituted, split, merged, or replaced by garbage.
  - Examples include `1.18` becoming `1 1 8`, `10^{-3}` becoming `1 0^{-3}`, `N_2` becoming `\Nu_2`, `lx` becoming `1x`, a variable becoming `�` or `□`, or `Var` becoming `V a r`.
  - Include corrupted decimal points, Greek/Latin letter substitutions, wrong OCR lookalikes, repeated garbage characters, and damaged chemical symbols when the input itself supports the corruption.
  - Do not flag valid Unicode math, legitimate variable naming, harmless LaTeX spacing, or unfamiliar notation.
  - If an external citation, page number, question number, or word is inserted into a formula, use Formula_Extra_Content instead.
  - Do not use domain knowledge alone to replace one valid-looking symbol with another.

  **Formula_Unparseable** — Broken LaTeX, Markdown, HTML, or MathML syntax prevents a standard parser from reliably identifying or rendering the formula.
  - Flag unmatched mathematical `$` or `$$` delimiters, unmatched `\(`/`\)` or `\[`/`\]`, missing required braces, mismatched or unclosed `\begin{...}`/`\end{...}` environments, unfinished commands, or unclosed formula-related HTML/MathML tags.
  - Flag raw LaTeX without math delimiters only when the target is clearly Markdown math and the commands consequently appear as unrendered source text.
  - Distinguish mathematical dollar delimiters from currency such as `$100` and `$5.99`.
  - Do not mechanically compare delimiter counts without context. `\left(a,b\right]` may be a valid half-open interval, and `\left.` or `\right.` is a valid invisible delimiter.
  - Do not flag well-formed `align`, `aligned`, `cases`, matrices, normal `\\` line breaks, or mixed LaTeX/HTML that renders correctly.
  - If the formula parses but its internal organization is damaged, use Formula_Structure_Corruption. If it parses but is placed or broken across lines incorrectly, use Formula_Layout_Corruption.

  **Formula_Structure_Corruption** — A formula can be parsed or rendered, but its internal hierarchy or component relationships were damaged during extraction.
  - Flag a multiplicative term moved into an exponent, a numerator and denominator placed incorrectly, reversed or displaced sum/integral limits, an operator with the wrong scope, a subscript attached to the wrong base, or formula fragments placed in the wrong internal order.
  - Flag damaged matrix rows or columns, lost array separators, corrupted determinants, malformed piecewise branches, or a `cases` expression converted into an unrelated structure such as `\binom`.
  - Require visible extraction evidence: an internal contradiction with a nearby definition, an intact copy elsewhere in the input, obvious layout reconstruction artifacts, or an unmistakably displaced formula component.
  - Do not flag a structurally complete statement solely because it is mathematically false, such as `2+2=5`.
  - Do not flag unusual but valid notation, or rewrite a formula merely because a more familiar form exists.
  - A single misrecognized character belongs to Formula_Token_Corruption; a missing component belongs to Formula_Partial_Loss; a parser failure belongs to Formula_Unparseable.

  **Formula_Layout_Corruption** — Formula content and internal structure are intact and parseable, but their placement, line breaks, or inline/display relationship with surrounding text is damaged.
  - Flag an inline formula forced onto its own line when this breaks sentence flow or produces abnormal display.
  - Flag one continuous equation split into separate math blocks, especially when one block ends with a dangling `+`, `-`, or `=` and the next block continues it.
  - Flag a display formula flattened into prose, a formula number separated from its formula, a formula moved to the wrong paragraph, or formula explanations placed in the wrong order.
  - Do not flag intentional display equations, harmless wrapping, normal multiline `align`/`cases`/matrix layouts, or an inline formula that merely wraps visually because of page width.
  - If a line break damages the delimiters and prevents parsing, use Formula_Unparseable. If unrelated text is inserted, use Formula_Extra_Content.

  **Formula_Extra_Content** — Non-formula content is inserted into a formula or attached to it in a way that forms an incorrect expression.
  - Flag a stray standalone `latex` token left after formulas, or leaked extractor/HTML markers.
  - Flag page headers, page footers, journal titles, author names, page numbers, dates, footnotes, captions, navigation, or unrelated prose inserted into a formula or derivation.
  - Flag citation numbers fused into formula exponents or subscripts, such as citation `[42]` turning `cm^{-1}` into `cm^{-142}`.
  - Flag question numbers fused into results, such as the next item number `1` turning an unfinished exercise into `145 × 12 = 1`.
  - Do not flag normal `\tag{3}`, a citation placed cleanly after a closed formula, normal explanatory prose following a formula, legitimate `\text{if}`/`\text{otherwise}`, units, or a document that is genuinely discussing LaTeX.
  - If there is only a placement or line-break problem and no foreign content was inserted, use Formula_Layout_Corruption.

  **Normal formula patterns (DO NOT flag)**:
  - Mixing inline `$...$` and display `$$...$$` formulas.
  - Using `\begin{align}...\end{align}` within `$$...$$`.
  - Normal `\\` line breaks in alignment, cases, or matrix environments.
  - Correct formula-related HTML such as `<sub>x</sub>` and `<sup>2</sup>`.
  - Mixing LaTeX and HTML in web-extracted content when the expressions remain correct and readable.
  - Plain-text math such as `a^2 + b^2 = c^2` without LaTeX delimiters when the target format allows it and the expression is complete.
  - A prose passage that mentions a result without displaying its derivation.
  - An isolated recoverable OCR typo that does not materially affect formula meaning.

- **Table_Corruption**: A clearly intended table whose structure or essential data is damaged
  - Misaligned rows/columns: values can no longer be matched to the correct headers
  - Missing essential body data: only a title, header, note, or discussion remains where the text explicitly depends on table values
  - Truncated rows/columns or broken HTML/Markdown table markup that makes the table unreadable
  - Flattened extraction is BAD only when row/column relationships cannot be recovered from the text
  - Impact: Models cannot learn proper table representation

  ⚠️ **DO NOT flag**:
  - A simple key-value table without a header
  - A list, catalog, bibliography, or metadata block that was never a table
  - A flattened table whose field-value relationships remain clear and readable
  - A passage that references a table located outside the provided sample, unless the sample itself shows extraction loss

- **Code_Corruption**: Recognizable source code whose formatting or syntax tokens were damaged during extraction
  **Common corruption patterns**:
  - Missing code fence (` ``` `): a multi-line code block appears as prose and its boundaries are unclear
  - Lost indentation: Python/YAML code with all indentation stripped (flat lines)
  - Broken identifiers: spaces injected into tokens, e.g. `sys .argv`, `pts .append`, `i[ 0]`
  - Line numbers mixed with code, broken syntax highlighting markers
  - Keywords wrapped in inline backticks instead of a fenced block, e.g. `` `import` sys ``

  Example (BAD — indentation and identifiers destroyed):
  ```
  `import` sys
  pts = []
  for i in range( 1,len(sys .argv), 2):
  pts .append([int(sys .argv[i]), int(sys .argv[i +1])])
  ```
  Correct version would have a code fence, proper indentation, and no spaces inside `sys.argv`.

  - Impact: Teaches incorrect code syntax, broken tokenization patterns, and wrong indentation conventions

  ⚠️ **DO NOT flag**:
  - Short inline code, commands, identifiers, stack traces, or configuration fragments that remain readable
  - Code shown without a fence when indentation, boundaries, and syntax are still intact
  - Logical bugs, deprecated APIs, inefficient algorithms, or style violations; this label evaluates extraction corruption, not program correctness

**Key Question**: "Can the model learn proper formatting from this structure?"

---

## 2. Effectiveness (有效性)
**Impact**: Noise prevents models from learning meaningful semantic patterns.

**Check for**:
- **Garbled_Characters**: Encoding corruption, replacement characters, or anti-crawler character artifacts
  - Example (BAD): "â€™" (broken UTF-8), "□□□" (placeholder chars), "ï»¿" (BOM)
  - Threshold: >1% of characters are garbled
  - Impact: Corrupts token distributions

  ⚠️ **DO NOT flag**:
  - Valid Unicode or a language written in an unfamiliar script (e.g., Cyrillic, Arabic, Greek, CJK)
  - Accented letters, mathematical symbols, bullets, checkboxes, or normal document glyphs used meaningfully
  - A few OCR spelling errors when the text remains readable
  - Mixed-language text by itself; require objective character corruption

- **Words_Stuck**: Missing spaces break tokenization
  - Example (BAD): "Thequickbrownfoxjumpsoverthelazydog"
  - Threshold: >1% of text has word boundaries missing
  - Impact: Wrong subword tokenization patterns

  ⚠️ **DO NOT flag**:
  - URLs, email addresses, file paths, code, formulas, hashtags, product IDs, or intentionally concatenated identifiers
  - Normal compounds or languages that do not conventionally separate words with spaces
  - One isolated missing space; require repeated or substantial boundary loss

- **Lack_Punctuation**: Sentence boundaries unclear
  - Example (BAD): "I like apples they are red also I like oranges"
  - Impact: Models cannot learn sentence segmentation

  Apply only to continuous natural-language prose with systematic missing sentence separators, such as at least three consecutive sentence boundaries missing or a substantial passage rendered as one run-on sentence.

  ⚠️ **DO NOT flag**:
  - Titles, headings, bullet lists, book lists, catalogs, bibliographies, search queries, navigation, metadata, tables, or short fragments
  - Line-separated items that are not intended to be complete sentences
  - Poor grammar, machine-translated prose, topic incoherence, or web-page aggregation when punctuation is not the primary defect
  - One isolated missing comma or period

**Key Question**: "Would a human find this readable and coherent?"

---

## 3. Similarity (相似性)
**Impact**: Repetitive content reduces training efficiency and causes memorization.

**Check for**:
- **Duplication**: Excessive exact or near-exact repetition within the sample that dominates the useful content
  - Example (BAD): "I like blue. I like blue. I like blue. I like blue..." (>30% duplicate)
  - Threshold: Same sentence/phrase repeats >5 times OR duplicate ratio >30%
  - Impact: Over-represents certain patterns

  ⚠️ **DO NOT flag**:
  - Necessary terminology, names, labels, citations, section titles, or short structural tokens repeated in otherwise diverse content
  - Repeated headers/footers that are brief and do not dominate the sample
  - Related statements that discuss the same topic but convey different information

**Key Question**: "Does this text provide diverse training signal?"

---

## 4. Security (安全性)
**Impact**: Harmful content should not be learned by models.

**Check for**:
- **Politics**: Review politically sensitive content from a Chinese perspective. Check **only** these five aspects:
  1. Content that undermines the state political system or socialist system
  2. Content that harms China's national security, national interests, reputation, or image
  3. Content that undermines national unity, territorial integrity, or social stability
  4. Content that promotes terrorism or extremism
  5. Content that incites ethnic or racial hatred and discrimination

  Flag `Politics` only when the text explicitly advocates, supports, promotes, or incites one of the five aspects above. Neutral mention, factual discussion, criticism of the harmful conduct, or content intended to prevent it must not be flagged. Do not expand `Politics` beyond these five aspects.

- **Prohibition**: Check **only** these four aspects:

  1. **porn** — Explicit sexual content
     - Direct or detailed descriptions of sexual acts or sexual organs, content clearly intended to cause sexual arousal, obvious sexual innuendo, or strongly vulgar sexual language.

  2. **violent** — Graphic violence or bloodshed
     - Clear and detailed descriptions of violent acts, bloody injury, cruel killing, torture, or other conduct emphasizing the brutality, pain, or cruelty of the act.

  3. **gamble** — Gambling-related content
     - Any content involving gambling activities, including promotion, inducement, advertisements, participation guidance, betting methods, odds, gameplay, mechanics, or explanations of gambling rules.
     - Gambling rules or gameplay explanations must be flagged even when presented as neutral or informational content. Only content whose purpose is to prohibit, prevent, or warn against gambling is excluded.

  4. **drug** — Promotion of illegal drugs or controlled substances
     - Content or advertisements that encourage, induce, facilitate, or promote illegal drug use, production, purchase, or sale, including operational usage guidance, purchasing channels, price information, or production instructions.
     - Drug names, slang, or indirect expressions require contextual judgment and must not be flagged solely by keyword matching.

  Flag `Prohibition` when the text clearly matches at least one of the four aspects above and is itself promoting, depicting, inducing, facilitating, or, for gambling, explaining the harmful content. Do not flag legitimate news, education, science communication, medical or pharmaceutical information, legal material, warnings, weapon introductions, or game introductions when the harmful content is presented for informational rather than promotional or exploitative purposes. This contextual exclusion does not apply to gambling rules, methods, odds, mechanics, or participation guidance; those must be flagged unless the content explicitly prohibits or warns against gambling.

**Key Question**: "Is this content safe for model training?"

---

# Evaluation Principles

1. **Evidence First**: Cite a concrete defect visible in the input; never infer a defect solely from topic, language, or document genre
2. **Focus on Training Impact**: Only flag issues that significantly harm LLM learning
3. **Severity Matters**: Minor typos and isolated defects are acceptable; systemic or meaning-destroying corruption is not
4. **Context Awareness**: Interpret characters and formatting according to the detected language and content type
5. **Threshold-Based**: Enforce stated thresholds (>1%, >30%, >5 times); do not treat them as optional guidance
6. **Primary Cause**: The reason must explain why the selected label, rather than another label, is the dominant defect

---

# Workflow

1. **Detect Context**: Identify the language/script and whether the input is prose, list, table, code, math, metadata, or mixed content
2. **Quick Scan**: Is the text generally readable, coherent, and structurally recoverable?
3. **Collect Evidence**: Locate explicit defects and check label-specific thresholds and exclusions
4. **Identify Defects**: Collect every distinct label whose threshold is independently met.
5. **Verify Impact**: Would this issue meaningfully harm model training rather than merely reduce stylistic quality?
6. **Assign Labels**:
   - Return one object per supported defect, each with score 0
   - If no defect is supported, return exactly one Good object with score 1
   - Type: 'Good' OR one of ['Completeness', 'Effectiveness', 'Similarity', 'Security']
   - Name: Specific error type (see above)
   - Reason: Brief explanation (1-2 sentences)

---

# Output Format
Return a non-empty JSON array only: [{"score": 0/1, "type": "", "name": "", "reason": ""}]

For defective text, include all independently supported labels. Do not include a Good object together with defect objects. Emit each label at most once.

Allowed score/type/name combinations:
- `1 / Good / None`
- `0 / Completeness / Formula_Missing`
- `0 / Completeness / Formula_Partial_Loss`
- `0 / Completeness / Formula_Token_Corruption`
- `0 / Completeness / Formula_Unparseable`
- `0 / Completeness / Formula_Structure_Corruption`
- `0 / Completeness / Formula_Layout_Corruption`
- `0 / Completeness / Formula_Extra_Content`
- `0 / Completeness / Table_Corruption`
- `0 / Completeness / Code_Corruption`
- `0 / Effectiveness / Garbled_Characters`
- `0 / Effectiveness / Words_Stuck`
- `0 / Effectiveness / Lack_Punctuation`
- `0 / Similarity / Duplication`
- `0 / Security / Politics`
- `0 / Security / Prohibition`

Never invent a label or pair a name with the wrong type.

The `reason` must cite a short concrete example or measurable pattern from the input. Do not use vague statements such as "low quality" or "unreadable" without evidence.

# Examples

**Example 1 (Good - Simple)**:
Input: "The Pythagorean theorem states that $a^2 + b^2 = c^2$ for right triangles."
Output: [{"score": 1, "type": "Good", "name": "None", "reason": "Clear, well-formatted text with proper LaTeX"}]

**Example 1.5 (Good - Complex Academic)**:
Input: "Friedmann equation:
$$
\begin{align*}
\left(\frac{\dot{a}}{a}\right)^2 &= \frac{8\pi G}{3}\rho \\
H^2 &= H_0^2[\Omega_m(1+z)^3 + \Omega_\Lambda]
\end{align*}
$$
where $a$ is scale factor and $H$ is Hubble parameter."
Output: [{"score": 1, "type": "Good", "name": "None", "reason": "Well-formed multi-line equations with proper alignment"}]

**Example 1.6 (Good - Mixed HTML/LaTeX)**:
Input: "The eigenstate $\psi_n$ where <sub>n</sub> is quantum number and energy E<sup>2</sup> = m<sup>2</sup>c<sup>4</sup>"
Output: [{"score": 1, "type": "Good", "name": "None", "reason": "Normal mix of LaTeX and HTML tags from web content"}]

**Example 1.7 (Good - Valid Non-Latin Script)**:
Input: "Заседание состоялось 14 сентября 2017 года. Протокол был утверждён членами совета."
Output: [{"score": 1, "type": "Good", "name": "None", "reason": "Readable Russian prose in a valid Cyrillic script; unfamiliar script is not character corruption"}]

**Example 1.8 (Good - List Without Sentence Punctuation)**:
Input: "Required documents:
Passport
Proof of address
Application form
Payment receipt"
Output: [{"score": 1, "type": "Good", "name": "None", "reason": "A clear itemized list; list entries do not require sentence-ending punctuation"}]

**Example 2 (Bad - Entire Formula Missing)**:
Input: "The compound has the following structure: [Chemical Formula 1]. In Formula 1, R is hydrogen and n is an integer from 1 to 6."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Missing", "reason": "Only the placeholder '[Chemical Formula 1]' remains, while the following sentence defines variables from the absent structure"}]

**Example 2.1 (Bad - Partial Formula Loss)**:
Input: "The transition model maps states and actions as $T(s,a,s') [0,1]$, where the output is a probability."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Partial_Loss", "reason": "The mapping relation between T(s,a,s') and [0,1] is missing its arrow"}]

**Example 2.2 (Bad - Formula Token Corruption)**:
Input: "Nitrogen adsorption was measured using $\Nu_2$. The same gas is referred to as nitrogen throughout the paragraph."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Token_Corruption", "reason": "The nitrogen symbol N is OCR-corrupted into the Greek command \\Nu in the formula \\Nu_2"}]

**Example 2.3 (Bad - Unparseable Formula)**:
Input: "The formula $x^2 + y^2 is broken here $$a = b$$$"
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Unparseable", "reason": "The first mathematical $ delimiter is never closed and an extra $ remains at the end"}]

**Example 2.4 (Bad - Internal Formula Structure)**:
Input: "Fuel flow is the product of coefficient μ, nozzle area A, and the square root of pressure difference. The extracted formula is $Q=\mu A^{\sqrt{\Delta P}}$."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Structure_Corruption", "reason": "The nearby definition says the square-root term is multiplied by A, but the extracted formula incorrectly places it in A's exponent"}]

**Example 2.5 (Bad - Formula Layout)**:
Input: "The value is\n$x+y$\nin this case."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Layout_Corruption", "reason": "The inline formula is forced onto a separate line, breaking the surrounding sentence flow"}]

**Example 2.6 (Bad - Extra Content Around Formula)**:
Input: "The normalized direction is $x_l=(x_2-x_1)/R$ latex and the next paragraph continues the derivation."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Extra_Content", "reason": "A stray extractor token 'latex' remains attached to the formula"}]

**Example 2.7 (Bad - Independent Formula Defects)**:
Input: "The material structure is: [Formula 1]. Separately, the energy equation is $E=mc^2$ latex."
Output: [{"score": 0, "type": "Completeness", "name": "Formula_Missing", "reason": "The material structure is absent and only '[Formula 1]' remains"}, {"score": 0, "type": "Completeness", "name": "Formula_Extra_Content", "reason": "A separate valid energy formula is followed by the stray extractor token 'latex'"}]

**Example 3 (Bad - Garbled Characters)**:
Input: "The exported text contains broken symbols â€™ â€œ □□□ ï»¿ throughout the paragraph."
Output: [{"score": 0, "type": "Effectiveness", "name": "Garbled_Characters", "reason": "Repeated mojibake, placeholder squares, and a leaked BOM exceed the character-corruption threshold"}]

**Example 3.1 (Bad - Words Stuck)**:
Input: "The extraction removed spaces in many places: theexperimentwascompleted and theresultswererecorded before thesampleswerediscarded."
Output: [{"score": 0, "type": "Effectiveness", "name": "Words_Stuck", "reason": "Multiple word boundaries are missing across a substantial portion of the passage"}]

**Example 3.2 (Bad - Lack of Punctuation)**:
Input: "The experiment was completed the results were recorded the samples were discarded the laboratory was then closed"
Output: [{"score": 0, "type": "Effectiveness", "name": "Lack_Punctuation", "reason": "Continuous prose has at least three missing sentence boundaries, producing a long run-on passage"}]

**Example 4 (Bad - Similarity)**:
Input: "Blue is nice. Blue is nice. Blue is nice. Blue is nice. Blue is nice. Blue is nice."
Output: [{"score": 0, "type": "Similarity", "name": "Duplication", "reason": "Same sentence repeats 6 times, indicating low content diversity"}]

**Example 5 (Bad - Multiple Labels)**:
Input: "Thequickbrownfox. Thequickbrownfox. Thequickbrownfox. Thequickbrownfox. Thequickbrownfox. Thequickbrownfox."
Output: [{"score": 0, "type": "Effectiveness", "name": "Words_Stuck", "reason": "Word boundaries are missing in every repeated sentence"}, {"score": 0, "type": "Similarity", "name": "Duplication", "reason": "The same sentence repeats 6 times"}]

---

# Input content to evaluate:

"""
