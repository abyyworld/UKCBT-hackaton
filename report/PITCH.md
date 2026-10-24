# 3-minute pitch script (6 slides: `report/Pitch_3min_BatmanAndRobin.pptx`)

About 420 words at a calm pace. Times are cumulative. Say "flagged for review", never "criminal".

## Slide 1, title (0:00 to 0:15)
"We're Batman and Robin. We gave every one of the 40,101 test transactions a suspicion score, and our one-line
finding is this: on this ledger, laundering gives itself away by its tempo."

## Slide 2, tempo chart (0:15 to 0:55)
"Almost every step of a planted scheme is the first payment between two wallets. But so are half of all normal
payments, so novelty alone is useless. What separates them is speed for the shape. Easy laundering moves a step
about every minute. Normal businesses that build the same shapes, like marketplaces and payroll, move about every
ten minutes. And the medium schemes are deliberately slowed to about eighteen. Amounts, fees and currencies look
almost the same. So we built features that describe whole motifs, not single payments: episodes of linked
payments, how much value each hop keeps, and whether the wallets do anything else during the scheme."

## Slide 3, results (0:55 to 1:30)
"We use two stages. A first model scores each payment; a second one lets each payment see the scores of the
payments linked to it, so a whole scheme lifts its weakest steps. On top, a specialist learns only the difference
between real motifs and their normal look-alikes. Out-of-fold average precision went from 58.6 to 67.5, and every
idea had to beat the noise on several seeds. Many good-sounding ideas did not, and we dropped them."

## Slide 4, flagged example (1:30 to 2:05)
"Here is our rank-one transaction. One wallet splits about 127 dollars across six brand-new wallets in under two
minutes, and all six pay the same collector within the next two. Our rank-sixteen case does the same but converts
each branch into euros or pounds and back, keeping 98 to 100 percent of the value. And when the collector later
pays an old contact, that payment ranks around 36060th: we score the motif, not the wallet."

## Slide 5, the hub we leave alone (2:05 to 2:30)
"This is the busiest wallet in the data: 316 payments, 247 counterparties. Any volume rule would flag it. We
don't: it moves at a steady business pace, never passes money straight on, and none of its payments reach our
top one percent. Volume is not evidence."

## Slide 6, limits (2:30 to 3:00)
"Honest limits. This is synthetic data, so a high score proves nothing about real crime. The hard relay cases are
built from the same template as twenty thousand normal ones, and about twenty signals we tried stayed at chance.
The test week also differs from training. Everything reproduces with one command in about eight minutes. Thank
you."

## If asked
- **Why not catch the relays?** Every signal we tried, at payment and whole-scheme level, scored AUC 0.57 or less.
- **Did you use labels as features?** No. The family column only chooses which rows the look-alike model trains on.
- **AI use?** Listed in `ATTRIBUTION.md`: AI coding assistants helped write code and run experiments; every number
  comes from our own runs.
