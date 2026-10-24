// Builds report/DecodeDetect_BatmanAndRobin.pptx (9 slides) from the numbers in report/SLIDES.md and EXAMPLES.md.
//   NODE_PATH=<dir with pptxgenjs> node report/make_deck.js [path/to/apply_theme.js]
// Final scores are set in FINAL below once the last run is uploaded.
const pptxgen = require("pptxgenjs");

const FINAL = { oof: "67.5", lb: "69.2", label: "v17, best so far" };

const THEME = {
  name: "Decode and Detect",
  headFontFace: "Cambria",
  bodyFontFace: "Calibri",
  colors: {
    dk1: "1B2433", lt1: "FFFFFF", dk2: "3A4659", lt2: "EEF2F6",
    accent1: "E8A33D", accent2: "2F7F86", accent3: "C8553D", accent4: "6B7A90", accent5: "A9BCD0", accent6: "3D5A80",
    hlink: "2F7F86", folHlink: "6B7A90",
  },
};
const HEX = THEME.colors;

const pres = new pptxgen();
pres.layout = "LAYOUT_16x9";
pres.title = "Decode and Detect: catching laundering by its tempo";
pres.author = "Team Batman and Robin";
pres.theme = { headFontFace: THEME.headFontFace, bodyFontFace: THEME.bodyFontFace };
const C = pres.SchemeColor;

pres.defineSlideMaster({
  title: "DARK",
  background: { color: C.text1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.6, y: 1.5, w: 8.8, h: 1.5, fontSize: 40, bold: true,
      color: C.background1, valign: "bottom" }, text: "" } },
    { placeholder: { options: { name: "body", type: "body", x: 0.6, y: 3.15, w: 8.8, h: 1.2, fontSize: 18,
      color: C.accent5, valign: "top" }, text: "" } },
  ],
});
pres.defineSlideMaster({
  title: "CONTENT",
  background: { color: C.background1 },
  objects: [
    { placeholder: { options: { name: "title", type: "title", x: 0.5, y: 0.3, w: 9.0, h: 0.75, fontSize: 28, bold: true,
      color: C.text1, valign: "middle" }, text: "" } },
    { text: { text: "Batman and Robin  |  Decode and Detect  |  flagged for review, synthetic data",
      options: { x: 0.5, y: 5.2, w: 7.5, h: 0.3, fontSize: 10, color: C.accent4, isTextBox: true, margin: 0 } } },
  ],
  slideNumber: { x: 9.0, y: 5.2, w: 0.5, h: 0.3, fontSize: 10, color: C.accent4, align: "right" },
});

function card(slide, x, y, w, h, title, body, opts = {}) {
  slide.addShape(pres.shapes.ROUNDED_RECTANGLE, { x, y, w, h, rectRadius: 0.08, fill: { color: opts.fill || C.background2 },
    line: { color: opts.fill || C.background2 }, objectName: `card ${title}` });
  slide.addText([
    { text: title, options: { bold: true, fontSize: opts.titleSize || 16, color: opts.titleColor || C.text1, breakLine: true } },
    { text: body, options: { fontSize: opts.bodySize || 14, color: opts.bodyColor || C.text2 } },
  ], { x: x + 0.15, y: y + 0.1, w: w - 0.3, h: h - 0.2, valign: "top", isTextBox: true, margin: 0, paraSpaceAfter: 4 });
}

function stat(slide, x, y, w, big, label, color) {
  slide.addText(big, { x, y, w, h: 0.75, fontSize: 40, bold: true, color: color || C.accent2, fontFace: THEME.headFontFace,
    isTextBox: true, margin: 0 });
  slide.addText(label, { x, y: y + 0.78, w, h: 0.55, fontSize: 13, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
}

// 1. Title
pres.addSection({ title: "Opening" });
let s = pres.addSlide({ masterName: "DARK", sectionTitle: "Opening" });
s.addText("Catching laundering by its tempo", { placeholder: "title" });
s.addText("Decode and Detect, UKCBT x XRPL AI Hack, 24 October 2026\nTeam Batman and Robin", { placeholder: "body" });
s.addNotes("We score every transaction in a synthetic XRP-style ledger for how suspicious it is. Our one-line story: on this data, laundering gives itself away by how fast money moves through brand-new wallet pairs, and the hard part is telling it apart from normal businesses that build the same shapes.");

// 2. Problem and data
pres.addSection({ title: "Problem" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Problem" });
s.addText("A week of payments, one suspicion score each", { placeholder: "title" });
stat(s, 0.5, 1.3, 2.1, "40,101", "test transactions to rank, one score each");
stat(s, 2.75, 1.3, 2.1, "7 days", "of synthetic ledger activity per split");
stat(s, 5.0, 1.3, 2.1, "3.1%", "of training transactions are suspicious");
stat(s, 7.25, 1.3, 2.25, "AP", "score = average precision: real cases at the top of the review queue", C.accent1);
card(s, 0.5, 3.15, 9.0, 1.15, "What we cleaned",
  "One row per transaction (1 to 3 transfer legs). XRP drops to XRP, assets keyed by issuer and symbol, every payment valued in USD at the hour's reference price. Suspicious cases: 64% easy, 18% medium, 18% hard.");
s.addNotes("We rank transactions, not wallets. Average precision rewards putting real cases at the top of an analyst's queue, so only the order matters. The cleaning follows the organiser traps: drops, issuer plus symbol, USD at the hour.");

// 3. Tempo
pres.addSection({ title: "Insight" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Insight" });
s.addText("Laundering gives itself away by its tempo", { placeholder: "title" });
s.addChart(pres.charts.BAR, [{ name: "Minutes between steps", labels: ["Easy laundering", "Normal look-alike", "Medium laundering"],
  values: [1, 10, 18] }], {
  x: 0.5, y: 1.2, w: 5.2, h: 3.8, barDir: "bar", showTitle: true, title: "Typical minutes between steps (train, median)",
  titleFontSize: 13, titleColor: HEX.dk2, titleFontFace: "+mn-lt", showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 13, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", chartColors: [HEX.accent1, HEX.accent2, HEX.accent1],
  catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent4, catAxisLabelFontSize: 13, valAxisLabelFontSize: 11,
  catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt", valGridLine: { color: "E3E8EE", size: 0.5 },
  catGridLine: { style: "none" }, showLegend: false, valAxisMaxVal: 20, catAxisOrientation: "maxMin",
});
s.addText([
  { text: "98% of planted steps are brand-new wallet pairs, but so are 55% of normal payments.", options: { bullet: true, breakLine: true } },
  { text: "Normal businesses build the same shapes at a business pace: about 10 minutes per step.", options: { bullet: true, breakLine: true } },
  { text: "Easy laundering is far faster; medium laundering is deliberately slower.", options: { bullet: true, breakLine: true } },
  { text: "Amounts, fees, assets and hour of day look almost identical.", options: { bullet: true } },
], { x: 6.0, y: 1.3, w: 3.5, h: 3.6, fontSize: 14, color: C.text2, valign: "top", isTextBox: true, paraSpaceAfter: 8 });
s.addNotes("This is the key finding. Novelty alone is not enough because half of normal payments are first-time pairs too. What separates laundering is unusual tempo for the shape: easy schemes step about every minute, normal look-alikes every ten, and medium schemes are deliberately stretched to about eighteen.");

// 4. Features
pres.addSection({ title: "Method" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Five feature families, all inside one week", { placeholder: "title" });
const fams = [
  ["Local", "Amount in USD, conversions, how busy sender and receiver are from 2 minutes to a day, first-time pair or not."],
  ["Episodes", "Payments linked through a shared wallet form an episode; its size, span and step gaps describe the motif."],
  ["Scheme isolation", "One actor runs a scheme's wallets, so they make no other payments during it. Normal wallets do."],
  ["Patterns", "Cycles, reach, scatter-gather paths and dense blocks among first-time pairs over 2 and 6 hours."],
  ["Motif shape", "How much value pass-through wallets keep (laundering: 98 to 100%) and how regular the hops are."],
  ["No labels used", "Labels, scenario, family and difficulty never enter as inputs. Everything is computed per split."],
];
fams.forEach(([t, b], i) => {
  const col = i % 3, row = Math.floor(i / 3);
  card(s, 0.5 + col * 3.05, 1.25 + row * 1.95, 2.85, 1.75, t, b,
    i === 5 ? { fill: C.text1, titleColor: C.accent1, bodyColor: C.background1 } : {});
});
s.addNotes("Every feature is computed inside a single week and never reads labels. Episodes and patterns describe whole motifs rather than single payments; isolation and shape are what separate real schemes from businesses with the same structure.");

// 5. Model
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Method" });
s.addText("Two stages and a look-alike specialist", { placeholder: "title" });
const steps = [["Stage 1", "LightGBM on all features, 2 seeds"], ["Look-alike model", "trained only on motif scenarios: real vs normal"],
  ["Stage 2", "each payment sees the scores of payments linked to it"], ["Score", "one rank per transaction"]];
steps.forEach(([t, b], i) => {
  const x = 0.5 + i * 2.35;
  card(s, x, 1.3, 2.0, 1.55, t, b, i === 3 ? { fill: C.accent1, titleColor: C.text1, bodyColor: C.text1 } : {});
  if (i < 3) s.addShape(pres.shapes.RIGHT_ARROW, { x: x + 2.03, y: 1.88, w: 0.29, h: 0.4, fill: { color: C.accent4 },
    line: { color: C.accent4 }, objectName: `arrow ${i}` });
});
s.addText([
  { text: "Grouped 5-fold cross-validation: every planted scenario stays inside one fold.", options: { bullet: true, breakLine: true } },
  { text: "Single-seed scores move by about 3 points, so ideas were kept only if they helped on several seeds.", options: { bullet: true, breakLine: true } },
  { text: "The family column only chooses the look-alike model's training rows; it is never an input.", options: { bullet: true, breakLine: true } },
  { text: "No single feature shifts between train and test (largest single-feature AUC 0.55).", options: { bullet: true } },
], { x: 0.5, y: 3.1, w: 9.0, h: 1.95, fontSize: 14, color: C.text2, valign: "top", isTextBox: true, paraSpaceAfter: 6 });
s.addNotes("Stage 2 is collective scoring: if most of a scheme scores high, its missing first and last steps get pulled up. The look-alike model learns only from planted scenarios and their normal twins, which is where most of our false positives were.");

// 6. Results
pres.addSection({ title: "Results" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Results: each idea had to earn its place", { placeholder: "title" });
s.addChart(pres.charts.BAR, [
  { name: "Out-of-fold AP", labels: ["Local only", "+ episodes, stage 2", "+ patterns, shape", "+ look-alike model"], values: [58.6, 61.6, 63.8, 67.5] },
  { name: "Leaderboard", labels: ["Local only", "+ episodes, stage 2", "+ patterns, shape", "+ look-alike model"], values: [0, 67.3, 68.5, 69.2] },
], {
  x: 0.5, y: 1.15, w: 6.1, h: 3.9, barDir: "col", barGapWidthPct: 60, showValue: true, dataLabelPosition: "outEnd",
  dataLabelFontSize: 12, dataLabelColor: HEX.dk1, dataLabelFontFace: "+mn-lt", dataLabelFormatCode: '0.0;;""',
  chartColors: [HEX.accent2, HEX.accent1], valAxisMinVal: 50, valAxisMaxVal: 75, showLegend: true, legendPos: "b",
  legendFontSize: 12, legendFontFace: "+mn-lt", legendColor: HEX.dk2, catAxisLabelColor: HEX.dk2, valAxisLabelColor: HEX.accent4,
  catAxisLabelFontSize: 11, valAxisLabelFontSize: 11, catAxisLabelFontFace: "+mn-lt", valAxisLabelFontFace: "+mn-lt",
  valGridLine: { color: "E3E8EE", size: 0.5 }, catGridLine: { style: "none" },
});
stat(s, 6.9, 1.25, 2.6, FINAL.lb, `final leaderboard score (${FINAL.label})`, C.accent1);
stat(s, 6.9, 2.6, 2.6, FINAL.oof, "out-of-fold AP, grouped 5-fold");
s.addText("Dropped after testing: relay features, slow-window counts, closed squares, a medium-only model, pseudo-labelling, value matching.",
  { x: 6.9, y: 3.95, w: 2.6, h: 1.1, fontSize: 11, color: C.accent4, valign: "top", isTextBox: true, margin: 0 });
s.addNotes("Out-of-fold rose from 58.6 to 67.5. The leaderboard runs higher than out-of-fold, but the look-alike model gained 3.7 points offline and only 0.6 on the leaderboard: the test set contains variations that are not in training, which we discuss under limits.");

// 7. Examples
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("Flagged for review: split, hop, merge in minutes", { placeholder: "title" });
s.addShape(pres.shapes.OVAL, { x: 0.6, y: 2.55, w: 0.75, h: 0.75, fill: { color: C.accent1 }, line: { color: C.accent1 }, objectName: "source" });
s.addShape(pres.shapes.OVAL, { x: 3.65, y: 2.55, w: 0.75, h: 0.75, fill: { color: C.accent3 }, line: { color: C.accent3 }, objectName: "collector" });
for (let i = 0; i < 6; i++) {
  const y = 1.3 + i * 0.55;
  s.addShape(pres.shapes.OVAL, { x: 2.2, y, w: 0.38, h: 0.38, fill: { color: C.accent2 }, line: { color: C.accent2 }, objectName: `mid ${i}` });
  s.addShape(pres.shapes.LINE, { x: 1.35, y: 2.92, w: 0.85, h: y + 0.19 - 2.92, line: { color: C.accent4, width: 1 }, objectName: `in ${i}` });
  s.addShape(pres.shapes.LINE, { x: 2.58, y: y + 0.19, w: 1.07, h: 2.92 - (y + 0.19), line: { color: C.accent4, width: 1 }, objectName: `out ${i}` });
}
s.addText("Rank 1 of 40,101   5736BA1F...\n$127 split across 6 new wallets in 1.7 min; all 6 pay one collector within 2 min",
  { x: 0.5, y: 4.55, w: 4.2, h: 0.6, fontSize: 12, color: C.text2, isTextBox: true, margin: 0, valign: "top" });
card(s, 4.95, 1.25, 4.55, 1.5, "Rank 4   8AA1DCA1...",
  "Same split and merge, but every branch converts USD into EUR or GBP and back. 98 to 100% of the value survives each hop, all within 10 minutes.");
card(s, 4.95, 2.95, 4.55, 1.5, "Rank 5   41EF2ECA...",
  "Four new wallets, four assets, merged in GBP in 2 minutes. The collector's next payment, to an old contact, ranks 17,189: we score the motif, not the wallet.");
s.addText("Full hashes in report/EXAMPLES.md. A high score means flagged for review, not proof of wrongdoing.",
  { x: 4.95, y: 4.6, w: 4.55, h: 0.5, fontSize: 11, color: C.accent4, isTextBox: true, margin: 0, valign: "top" });
s.addNotes("Our top case is a textbook scatter-gather: one wallet splits about 127 dollars across six brand-new wallets in under two minutes, and all six pay the same collector within the next two. The third example shows the model ignores the collector's ordinary follow-on payment.");

// 8. Hub
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Results" });
s.addText("The busiest wallet, deliberately not flagged", { placeholder: "title" });
stat(s, 0.5, 1.3, 2.1, "316", "payments in the test week");
stat(s, 2.75, 1.3, 2.1, "247", "different counterparties, 88% of them new");
stat(s, 5.0, 1.3, 2.1, "7 of 7", "days active, a steady 20 minutes apart");
stat(s, 7.25, 1.3, 2.25, "1,975", "its best rank; none of its payments in our top 1%", C.accent1);
card(s, 0.5, 3.2, 9.0, 1.7, "Why we leave it alone   rnpfKt4h...",
  "By raw fan-in and fan-out it is the most extreme wallet in the data: exactly the exchange or payroll hub the brief warns about. But its new pairs arrive at a steady business pace, money is not passed straight on, and its counterparties do not form closed groups. Volume is not evidence.");
s.addNotes("Hubs look anomalous by any count. We want the model to score behaviour, not size: steady tempo, no pass-through and open, unrelated counterparties are what normal businesses look like.");

// 9. Limits and disclosure
pres.addSection({ title: "Close" });
s = pres.addSlide({ masterName: "CONTENT", sectionTitle: "Close" });
s.addText("Limits, next steps and how we built it", { placeholder: "title" });
card(s, 0.5, 1.2, 4.4, 3.8, "Limits and next steps",
  "Synthetic data: a high score here proves nothing about real crime.\n\nHard relay cases (16% of suspicious) are built from the same template as 19,800 normal ones; about 20 signals we tried stayed at chance. 25 labels are random by design.\n\nTest contains variations absent from training (amounts look less even). Next: validation that holds out whole batches, more labels for slow schemes.",
  { bodySize: 13 });
card(s, 5.1, 1.2, 4.4, 3.8, "AI use and reproducibility",
  "AI coding assistants (Claude Code and others, listed with models in ATTRIBUTION.md) helped write code and run experiments; every score comes from our own runs.\n\npython run.py rebuilds the upload file from raw data in about 8 minutes on a 4-core machine, fixed seeds, checked twice from an empty cache.\n\nMIT licence. No outside data.",
  { bodySize: 13, fill: C.text1, titleColor: C.accent1, bodyColor: C.background1 });
s.addNotes("Being honest about limits: the hard relays and random labels cap any honest model at roughly 82 average precision on train. The test week also differs from training, which is why our offline gains did not all transfer. Everything is reproducible with one command.");

(async () => {
  const out = "report/DecodeDetect_BatmanAndRobin.pptx";
  await pres.writeFile({ fileName: out });
  if (process.argv[2]) {
    const { applyTheme } = require(process.argv[2]);
    await applyTheme(out, THEME);
  }
  console.log("wrote", out);
})();
