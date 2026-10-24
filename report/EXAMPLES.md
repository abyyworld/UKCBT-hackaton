# Flagged for review: examples from the test predictions

From `submissions/final.csv` (run.py at e896834, the final five-model blend). Ranks are out of 40,101 test transactions.
"Flagged for review" only: this is synthetic data, and a high score is not evidence that anyone did anything wrong.
Times are minutes from the first payment shown. "New pair" means the first payment between those two wallets in
the week.

## 1. Scatter-gather, under 4 minutes (rank 1)
`5736BA1F8EA4495FB06B198310ABBB052E0CE126A3C2056FBB153329531B083C`

`rNjNitRF…` pays six brand-new wallets $12 to $26 each (about $127 in total) within 1.7 minutes. Between minute 2.0
and 3.8, each of the six pays the same collector `rUYRMFCQ…` almost the same amount (about $132 in total). Twelve
payments, all new pairs, all USD. The six middle wallets do nothing else in this window. Ten of the twelve payments
rank in our top 17.

## 2. Scatter-gather with currency hopping (rank 16)
`8AA1DCA1DE9BD69A302F665CB5A3FD5FD802E97F70A86A2D7D795819BBA50026`

`rhFAWFuX…` sends USD to six new wallets over 6 minutes, converted on the way into EUR, GBP or kept as USD. From
minute 6.9 to 10.3 each middle wallet converts back and pays USD to one collector, `rBDWR1c6…`. Each hop keeps 98% to
100% of the value. Money that splits, changes currency and comes back together within 10 minutes.

## 3. Same shape, and what we do not flag (rank 93)
`41EF2ECA2F68D3D73A0ACC44F1EB03DAB04AA7485DD3032F2E171D05F665A23E`

`r3FJPbfv…` (also a middle wallet in example 2) splits about $281 across four new wallets in 1.6 minutes, with four
different output assets (GBP, USD, XRP, EUR). Within 2 minutes all four pay `rHp1m9XL…` in GBP. Ten minutes later
`rHp1m9XL…` pays an old counterparty it already knew: that payment ranks 36060, so the model scores the motif and
not every payment the collector makes.

## Look-alike we leave alone: a busy hub
`rnpfKt4h6eKnfrhWNu8Jzkmfkb6R5RC7DU`, the busiest wallet in the test week: 316 payments, 247 different
counterparties, 88% of them new pairs, active on all 7 days, a steady median gap of 20 minutes between payments.
By raw fan-in and fan-out it is the most extreme wallet in the data, but none of its payments are in our top 1%
(best rank 1,204, median rank 32,657). Its new pairs arrive at a steady business pace, money is not passed straight
on, and its counterparties do not form closed groups. This is the exchange or payroll pattern the brief warns about.
