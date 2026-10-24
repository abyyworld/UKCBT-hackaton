# UKCBT × XRPL AI Hack: Decode & Detect

Hackathon: **24 October 2026**, IDEALondon, 69 Wilson Street. *Build a detection algorithm. Test your approach.*
Prize pool: £10,000 (£5,000 for 1st, £3,500 for 2nd, £1,500 for 3rd), awarded per team, including solo entries.
**All submissions close at 16:30 London time.**

Hosted by the UK Centre for Blockchain Technologies (UK CBT), in partnership with Ripple, the UCL Centre for
Blockchain Technologies, XRPL Commons, the Imperial College London Data Science Society, the King's College London
Blockchain Society and GirlsWhoML.

This brief draws on three sources: the two training decks in [`training/`](training/), the pre-hackathon training session
on 23 October 2026, and the welcome slides from the day in [`briefing/`](briefing/). When the data package arrives,
its `README.md` takes precedence over this brief.

## On the day

| Time | Activity |
|---|---|
| 09:30-10:00 | Arrival, check-in and meeting teammates |
| 10:00-10:30 | Welcome, challenge briefing and questions |
| 10:30-16:30 | Hacking, with technical support available |
| Around 13:00 | Working lunch |
| **16:30** | **All submissions close** |
| After 16:30 | Initial submission checks and verification, winners announcement and prizes, networking and drinks |

**Teams and participation**

- Compete individually or in a team of 1-5 people. Each person can belong to only one team.
- Every participant must be a student aged 18 or over.
- Every participant must have accepted the Luma registration, confirmed their competition registration, personally
  accepted the T&Cs and attended the mandatory online training.
- **Only eligible participants who are physically present at the venue can compete.** Check-in must be complete by **10:30**.
- Confirm the final team name, captain and members with the organisers.
- Each final team, including solo competitors, receives a **unique submission code**, which is needed to upload.
- If you're still looking for teammates, speak to Narmin Rahimli.

**Mentors:** for questions about the rules, the data, the submission process or anything else, speak to Jiahua Xu,
Walter Hernandez, Chantelle Obatolu, Narmin Rahimli or Alyssa Sorzano. Additional online help may also be available.

**Practical information**

- **Wi-Fi:** use Eduroam, or create an account on the Sky WiFi page with any email address and a password of your
  choice. The same login works on all your devices.
- **Technical help:** Alyssa Sorzano. **Event and team-registration help:** Narmin Rahimli.
- **Lunch:** served to the tables around 13:00.
- **Toilets:** unisex, downstairs, past the Event Space ("The Hub") and behind the second 4-person glass meeting
  room, labelled "Sandy".
- **Fire exits:** marked with the green symbol. The nearest are the main IDEALondon entrance by the café and one in
  the Event Space/Hub, near the screen on the left. If the alarm sounds, follow the main exit to the main building
  reception until you are out of the building. No fire drills are scheduled this month.
- **Photography and filming** will take place. Speak to an organiser if you have a concern or objection.

## What we have to build

> **Official brief (day-of slide 8):** Develop an algorithm that identifies suspicious payment patterns in our
> fictional blockchain dataset. Use transaction details, account relationships, timing and currency conversions to
> identify patterns associated with **simulated money laundering**, while limiting false positives. Assign every
> transaction in the supplied test dataset a **suspicion score between 0 and 1**. **Higher scores should indicate
> greater suspicion.** Predictions are evaluated against test labels held privately by the organisers.

In short, we need a **Kaggle-style anti-money-laundering (AML) model**. Given a synthetic ledger in the real
XRP Ledger (AWS/SonarX) format, it gives **every test transaction a suspicion score**. The job is to rank the
laundering transactions above the normal ones.

- The task is **money laundering**: tracing where money came from as it moves through many wallets.
  It is not transaction fraud, which asks whether a single payment should happen at all.
- The data is mostly about **layering**: money hopping, splitting, merging and converting between wallets.
- We score **transactions** (one row per `TRANSACTION_HASH`), not transfers or accounts.
- The model sits in the "monitor payments" step of the AML pipeline. It should put the real cases
  at the top of an analyst's review queue, and that is exactly what the metric measures.

## Submission

Upload a **predictions CSV and a code ZIP together by 16:30 London time on 24 October 2026**.
**A predictions CSV alone is not a complete final submission.**

### Predictions CSV (≤ 12 MiB)

One row for each of the **40,101** test transactions, with exactly these columns (download the template from the portal):

- `TRANSACTION_HASH`: the transaction identifier from the supplied test data.
- `score`: a numeric suspicion score between 0 and 1. Higher means more suspicious.
- `reason` (optional): the portal also accepts a short explanation per transaction.

Check before uploading:

- Every test transaction appears exactly once, with no missing, duplicate or extra transaction IDs.
- Every score is finite and between 0 and 1, and uses the full range (such as 0.82 or 0.07) rather than only 0 and 1.
- The column names and format exactly match the template.
- There is one prediction per **transaction**, even where a transaction contains several transfers.

### Code ZIP (≤ 8 MiB, ≤ 64 MiB uncompressed)

Leave out the data files and credentials. The portal stores the code for review but does not run it.

- **Predictions and results:** the test predictions CSV and the relevant validation results.
- **Code or notebook:** the code needed to generate the predictions.
- **Model and/or prompt set:** the relevant model details, prompts and settings, with model files or access
  instructions where applicable.
- **Run instructions:** the steps, dependencies, computing requirements and external services needed to reproduce
  the results, including the exact run command and seeds.
- **Slides or written report:** fewer than 10 slides, or a report of up to 4 pages in a legible font. It should cover
  the methodology, features, model or prompts, validation approach, results, interesting findings and limitations.
- **Team details:** team name, captain, final member list, and version or commit.
- **No access codes, credentials or secrets**, and no installed packages.

### How to submit

Use the submission portal at <https://hackathon.ukcbt.org/>.

1. Open the submission portal.
2. Sign in with the organiser-issued access code (the team's unique submission code) and download the data.
3. Upload the code ZIP and the predictions CSV together.
4. Check that the portal accepts the files and returns a score.
5. Review the score and the leaderboard. We can update the predictions before the deadline. The portal allows
   **200 attempts per day** for team **Batman and Robin**, and each attempt can have an optional run name.
6. Complete the supporting-material submission by following the portal instructions.

**Everything must be submitted by 16:30.** Submit early enough to fix any errors, and if something goes wrong,
contact Narmin Rahimli promptly. Pitch length will be confirmed on the day.

## Scoring

**Score = 100 × Average Precision (AP)** against private organiser labels on all 40,101 test transactions.
AP rewards ranking suspicious transactions ahead of legitimate ones across the whole test file, so a higher score is better.

Worked example: 5 transactions, 2 suspicious, ranked at positions 1 and 3. AP = (1/1 + 2/3) / 2 = 0.83, so the score is 83.

- **Only the order counts.** The scores don't need to be calibrated probabilities.
- **Avoid ties.** Equal scores are grouped into one block, so use continuous scores rather than 0/1 labels.
- **Diagnostics only:** ROC-AUC, and precision, recall and F1 in the top 1% (402 transactions). None of these
  count towards the score.
- The leaderboard shows the team's best completed score. Positions stay provisional until verification is
  complete, and final prize-entry selection needs organiser confirmation.
- **Winners:** the top three eligible, complete and verified entries by test score, with sufficient documentation
  and completed eligibility checks. If there is a tie between entries, the organisers will choose a fair resolution.

**Prizes and verification**

Provisional awards are announced on the evening of 24 October, after submissions close at 16:30. Before confirming them, the organisers check:

- participant and team eligibility;
- that all required submission materials are complete;
- compliance with the competition rules;
- whether they can reproduce the submitted predictions and results.

Rankings and awards stay provisional until these checks are complete. The final winners are announced within 14 days.

## The data

| | |
|---|---|
| Train | 40,131 transactions, labelled |
| Test | 40,101 transactions, unlabelled (these are scored) |
| Window | 7 days of activity per split |

| File | What is in it |
|---|---|
| `transactions.parquet` | One row per transaction: sender, fee, payment details as JSON |
| `transfers.parquet` | The money legs: 1 to 3 rows per transaction |
| `blocks.parquet` | Ledger records: index, hash, close time |
| `market_rates.csv` | Hourly reference USD price for each asset |
| `labels.csv` | Training split only: which transactions are suspicious |

The package also includes `payments.py`, which builds one edge per payment (who paid whom, how much in USD, and when),
and `baseline.py`, which produces a valid submission and a score to beat.

A transaction may contain several transfer records, but we **submit one prediction per transaction**.
Data and documentation are at <https://hackathon.ukcbt.org/>. The activity is fictional: performance on this challenge
says nothing about performance on real financial crime or the real XRP Ledger.

### Six traps for the first hour

1. **Score transactions, not transfers.** Join on `TRANSACTION_HASH` and output one row per transaction.
2. **XRP comes in drops.** For XRP rows, `SOURCE_VALUE` is in drops, and 1 XRP = 1,000,000 drops.
3. **An asset is issuer address + symbol.** Group by `TOKEN_ADDRESS` and `SYMBOL`, because USD from two issuers is two different assets.
4. **SendMax is a ceiling, not the cost.** The first transfer leg shows what was really paid, and `DELIVERED_AMOUNT` shows what arrived.
   The signed amount can also overstate what arrived (partial payments), so never use it as the value.
5. **Compare value in USD.** Use `market_rates.csv` for the hour that contains the payment.
6. **Some fields are JSON text.** Parse `EXTRA_TRANSACTION_FIELDS` and `DELIVERED_AMOUNT`.

Other ledger facts and traps:

- A wallet is just an `r…` address, so we judge behaviour, not identity.
- A conversion is a single transaction: the sender pays one asset and the receiver gets another through one or
  two liquidity providers, which produces 2 or 3 transfer rows.
- Every transaction burns 10 to 15 drops of XRP as a fee, including token payments.
- **Hubs look anomalous.** Exchange and payroll hubs have huge fan-in and fan-out, so give them their own class
  instead of treating them as outliers. One exchange address serves many customers, told apart by destination tag.
  If a tag is present, use the address and tag together as the customer.
- **Failed transactions.** A `tec…` result paid a fee but moved nothing. If the data has a result field, leave
  failed transactions out of value-flow edges but count them as a behaviour, and still score them.
- **Dust drowns the graph.** Set near-worthless payments aside when computing fan-in, fan-out and paths, but
  still give every one of them a score.
- **Ripple time.** Raw XRPL times count seconds from 1 Jan 2000. Add 946,684,800 to get Unix time before matching
  against the `market_rates.csv` hours.

### What is hiding in the labels

- Suspicious transactions by difficulty: **Easy 64% · Medium 18% · Hard 18%**.
- **Hard means history.** Hard cases span many payments with changing counterparties, so we have to look at whole
  flows over the week, not single amounts.
- There is plenty of normal activity that looks similar: shopping, peer-to-peer payments, exchange and payroll hubs,
  and legitimate treasury flows. Both normal and suspicious activity use conversions.
- **The test set is not the train set.** It contains variations that are absent from training, and a few cases
  are deliberately ambiguous. We must not overfit to the training patterns.
- A suspicious label describes the scenario's intent, not a rule for real life.

## Laundering shapes to look for

These pattern families come from IBM's AMLworld synthetic AML datasets (Altman et al., NeurIPS 2023).
Every one of them also occurs in legitimate business, and context is what tells them apart.

| Shape | How it works | Legitimate look-alike | Signals to compute |
|---|---|---|---|
| **Fan-in** | Many wallets, often mules, send modest amounts to one collector, which then forwards the total in one or a few payments | Shop, fundraiser, exchange deposit address | Distinct senders into one wallet in a short window; senders that are new or do nothing else; time from the last deposit to the exit payment; exit amount ≈ sum of the deposits |
| **Fan-out** | One wallet splits a large balance into many smaller payments, and the recipients often pass it on | Payroll, refunds, giveaways, exchange withdrawals | Distinct recipients per sender in a short window; how similar the payout amounts are; recipients that forward quickly; source funded just before |
| **Scatter-gather** | Money splits across middlemen, then merges again at one destination | A firm moving money between its own accounts, exchange rebalancing | Several two-hop paths between the same pair of wallets; value out of the source ≈ value into the destination; middle wallets with no other activity; short, similar hold times |
| **Gather-scatter** | A hub collects from many wallets and pays out to many others, acting like a mixing point | Exchange hot wallet, payment processor, payroll provider | Many senders and many receivers; little balance kept compared with the flow; short time from in to out; senders and receivers that never overlap |
| **Cycle** | Funds leave a wallet, pass through others and come back to it | Arbitrage, market making, refunds, treasury moves | A path back to the start within a time window; hop count and loop duration; value returned compared with value sent; assets changing along the way |
| **Layered stack** | Money passes through several rows of intermediary wallets | Payment processors, correspondent chains | Depth of the longest followable chain; middle wallets used once and then abandoned; value kept almost whole from layer to layer; similar delay at each layer |
| **Bipartite** | One group of wallets pays another, many to many, so every single pair stays small | Marketplace buyers and sellers, a company paying its suppliers | Dense payment blocks between two sets; groups with few outside contacts; many pairs active in the same window; similar amounts across pairs |
| **Irregular network** | Many partial routes with splits, merges and returns, matching no fixed template | Everyday peer-to-peer payments between friends, families and local traders | Size and density of the connected group; number of distinct routes between two wallets; value into the group compared with value out; how tightly the activity is packed in time |

The XRPL Commons deck adds two more shapes:

- **Peel chain:** a long path of fresh wallets, each taking a small cut.
- **Fresh-account batch:** new wallets funded by one parent minutes apart, used once, then deleted with `AccountDelete`.

Behaviours that raise a flag:

- **Rapid pass-through:** money leaves soon after it arrives and little is kept.
- **Hops with no purpose:** long chains through wallets that do nothing else.
- **Splitting and structuring:** one amount broken into many smaller payments, often sized to stay under thresholds.
- **Currency hopping:** converting through several assets to blur the trail.
- **Round-tripping:** funds come back to where they started, possibly in another asset.
- **Out of character:** activity that doesn't match the wallet's own earlier history.
- **Patterns over time:** a flow that only makes sense when you look at the whole week. Compute features over
  several window lengths, because a short window cuts slow flows in half. Flows at the edges of the 7-day split
  are also cut off.

**A pattern is not proof.** For example, a wallet that forwards 50 XRP minus a 15-drop fee 19 seconds later may be an
automated service. A loop of 40 payments of 0.00001 XRP around ten accounts may be a bot or a test, and its total is
worth almost nothing.

## Rules

**Data and tools**

- Use the supplied training and test files. Public documentation, research and reference data may support the work.
  (The pre-hackathon training said outside transaction or reference data needs approval, so check the full terms
  before using any.)
- **Disclose** any external data sources, APIs, models and services.
- Any programming language, framework or model type is allowed: rules, statistical methods, machine learning,
  graph algorithms or LLM-based approaches.
- AI coding assistants, existing libraries and pre-trained models are allowed. Identify the pre-existing components
  and explain what the team developed. The work being judged must be done during the hackathon.
- For any LLM use, disclose the model and provider, the relevant prompts and settings, and any reproducibility limitations.

**Fair participation**

- Do not access the private test labels, organiser-only files or other teams' private work.
- Do not fabricate results. Preserve third-party licences and attribution.
- Do not present synthetic results as evidence of wrongdoing by real people or accounts.

**Our work**

- The team keeps ownership, but the original submission materials must be released under an **approved open-source
  licence that permits commercial reuse**, whether or not the entry wins. Every team member accepts the terms.
- The team must understand its own solution, and the organisers must be able to reproduce the predictions. A run
  they can't reproduce may be retracted.

Full rules: <https://hackathon.ukcbt.org/terms/2026-10-21>

> **Which data counts (settled on the day).** The XRPL Commons deck (pp. 10, 29-35, 44-49) describes the
> challenge as pulling your own real Mainnet data, with no labels, and ranking accounts with Isolation Forest.
> The organisers' day-of briefing (slides 8-9) confirms the Decode & Detect version instead: the **supplied
> labelled synthetic dataset**, one score per test transaction, and no outside transaction data without approval.
> We use the Mainnet exercises only to learn the XRPL data model.

## Plan for the day

**Before we start:**

- [ ] Confirm the team (name, captain and members) and collect the submission code.
- [ ] Open the data, the documentation and the submission template.
- [ ] Check that we can access the submission portal.
- [ ] Choose the open-source licence (one that permits commercial reuse) and add a `LICENSE` file.
- [ ] Make a plan for development, validation and the final submission.

The organisers suggest this workflow: explore the supplied data, develop features, build an approach, validate on a
held-out part of the labelled training data, then generate a score for every test transaction. Our steps:

1. **Load and join** transactions, transfers and rates, and check the row counts.
2. **Build one edge per payment** (who paid whom, how much in USD, and when) using `payments.py`.
3. **Run the baseline** with `baseline.py` to get a valid file and a number to beat.
4. **Study the labels** by looking at the suspicious flows in training to see what makes them differ.
5. **Build features** from timing, amounts, graph shape and each wallet's history.
6. **Validate, then submit.** Hold out part of training and measure AP locally before spending uploads.
   - Keep transactions from the same flow on one side of the split, so the hold-out is not leaked.
   - Check that the ranking holds across different time slices of the week.
   - Plant a few altered fan-outs or peel chains and check that the model still finds them. The test set has
     variations that are not in training.
   - Report held-out AP only. Never report training AP, or the accuracy of a rule built from the labels, as the model's score.

Keep notes and a decision log as we go. The judges want to know *why* the model works, not only its score.

## The pitch

1. **Problem and data:** which behaviour we target, what data we used and how we cleaned it.
2. **Approach and evidence:** features, rules and model in plain words. Walk through two or three flagged cases
   with their `TRANSACTION_HASH`es, and show a look-alike we deliberately leave unflagged (such as an exchange or
   payroll hub), explaining why.
3. **Limits and next steps:** what we would do with more labels, time or data. Being honest about limits scores well.

Responsible analysis: say "flagged for review", show the evidence and say where every label came from.
Never call an address criminal or try to identify a person. A high score on synthetic data does not prove
a method works on real financial crime.

## Training materials

- [`briefing/`](briefing/): the welcome slides from the day (slides 2-6), covering the welcome, the hosts and partners,
  the agenda, teams and participation, and the mentors. Slides 7-16 (practical information, the challenge, the data,
  building your solution, scoring, the prediction file, the final submission, how to submit, the rules, and prizes and
  verification) are summarised in this README.
- [`training/Decode_and_Detect_AML_Training.pptx`](training/Decode_and_Detect_AML_Training.pptx): *Decode & Detect*,
  the AML and challenge training by Muhamed Retkoceri. It covers the patterns, the data, scoring and the rules.
- [`training/XRPL_AI_Hack_Training_UKCBT_October_2026.pdf`](training/XRPL_AI_Hack_Training_UKCBT_October_2026.pdf):
  *Laundering detection on the XRPL*, the XRPL fundamentals, data access and typologies training by
  [XRPL Commons](https://www.xrpl-commons.org/), licensed Creative Commons Attribution-ShareAlike (CC BY-SA).
  Credit XRPL Commons and keep the same licence when reusing or adapting it.
