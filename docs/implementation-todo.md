# Implementation Todo

## 1. Define the boundary contract

Complete at version `1.0.0`. See [boundary-contract.md](boundary-contract.md).

- [x] Define the normalized financial-state schema.
- [x] Define the enriched-request schema, including seller payment options.
- [x] Represent dates, recurrence, certainty, provenance, amendments, and exclusions.
- [x] Define schema invariants and deterministic validation rules.
- [x] Create representative valid and invalid fixture objects.

## 2. Build the deterministic decision engine

- [ ] Accept only a normalized financial state and enriched request.
- [ ] Generate the 90-day baseline cash-flow forecast.
- [ ] Calculate the maximum amount safe to pay on the request date.
- [ ] Calculate the earliest safe date for one full payment.
- [ ] Evaluate full, partial, installment, wait, and decline outcomes.
- [ ] Evaluate permitted spending reductions and cancellations.
- [ ] Rank safe plans using the required ordering.
- [ ] Generate and validate all required output fields.
- [ ] Test the engine against handcrafted fixtures and solved samples.

## 3. Build the normalizer incrementally

- [ ] Assemble each request's relevant profile and evidence.
- [ ] Normalize structured financial events and lifecycle links.
- [ ] Apply dated currency conversions deterministically.
- [ ] Extract and normalize facts from messages.
- [ ] Extract missing amounts and supporting facts from images.
- [ ] Detect recurring income and expenses from historical evidence.
- [ ] Resolve amendments, cancellations, duplicates, and conflicts.
- [ ] Record provenance and assumptions for every interpreted fact.
- [ ] Have the model review and sign off the final state object.
- [ ] Validate the signed object and retry with validation errors when necessary.

## 4. Integrate through vertical slices

- [ ] Handle a simple immediate full-payment case end to end.
- [ ] Add recurring and variable essential spending.
- [ ] Add pending and scheduled transactions.
- [ ] Add waiting and partial-payment scenarios.
- [ ] Add installment schedules and preference constraints.
- [ ] Add optional spending changes.
- [ ] Add messages, amendments, and conflicting evidence.
- [ ] Add image-derived amounts and foreign-currency events.
- [ ] Run all solved samples and classify failures by component.
- [ ] Run and validate the complete evaluation dataset.
