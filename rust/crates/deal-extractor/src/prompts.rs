//! Domain extraction prompts — ported from the TS `SECTION_PROMPTS` in
//! `server/src/handlers/deals/extraction/extract-section-assertions.ts`.
//! These encode the hard-won rules from 4 integration runs.
//!
//! The JSON contract itself is rendered from the dumped TS schema
//! (`contract::render_contract`), not written here, so it cannot drift.

pub const SYSTEM: &str = "\
You are a property-finance extraction analyst. You read deal documents \
(broker teasers, valuation reports, cash-flow models, planning docs) and \
extract structured facts into canonical JSON. You are precise, exhaustive, \
and never confuse distinct financial concepts.";

/// Round 0: whole-corpus extraction of the complete deal document.
/// The agent owns the full output — there is no merge step.
pub const EXTRACTION_PROMPT: &str = r#"Extract the complete property deal into canonical JSON from the documents below.

## FACILITIES
Extract EVERY facility mentioned: senior debt, mezzanine, whole loan, bridge, development loan, investment-term loan, VAT facility, equity, and any facility being repaid, refinanced, requested, proposed, or planned.

For each facility:
- id: stable slug (e.g. "bridge_facility", "dev_facility", "shojin_senior_repaid")
- name: as it appears in the documents
- instrument_type: one of debt_facility | ordinary_equity | preferred_equity | shareholder_loan | vendor_finance
- role: one of senior_debt | mezzanine_debt | junior_debt | whole_loan | developer_equity | preferred_equity | other
- scope_level: "deal" if it is part of the deal being underwritten, "project" if it is a future/project-scope facility, null if unclear
- loan_type: one of bridge | development | investment_term — NEVER free text like "finance requirement"
- commitment: { amount, currency } — `amount` is an exact DECIMAL STRING in absolute units ("25750000"), NEVER a JSON number and NEVER millions shorthand (not "25.75")
- term_months: number if stated — the stated EXPECTED facility term (e.g. "18-months expected"), never a period used only in an interest illustration, cash-flow example or extension scenario
- stated_interest: { amount, currency } — explicit interest amount for THIS facility if stated (e.g. "rolled-up interest £509,123", "mezzanine interest £X"), else null
- stated_redemption: { amount, currency } — the all-in repayment figure if stated (e.g. "loan including interest £5,699,659"), else null
- status: one of candidate | active | repaid | cancelled | superseded
- disposition: one of requested | existing | repaid | proposed
- lender_name: if named
- evidence: [{ document, page, quote }] citing the source

CRITICAL RULES:
1. Include repaid/refinanced facilities (e.g. existing Shojin debt being repaid) with status "repaid" and a matching fully_repaid/refinanced capital_event.
2. Include future/planned facilities (e.g. a development facility that will replace the bridge) with disposition "proposed" and scope_level "project".
3. Do NOT sum unrelated facilities — each facility is its own entry.
4. A bridge facility is short-term (≤12 months) acquisition/refinance financing; a development facility is longer-term construction financing.
5. "whole_loan" means one facility covering senior+mezzanine — check for "whole loan" wording before generic "loan"/"debt".
6. Equity facilities (developer equity, preferred equity) are NOT debt — instrument_type ordinary_equity/preferred_equity, role developer_equity/preferred_equity.

## FOCUS FACILITY
The focus is the NEW money being requested — not existing debt being repaid, not future planned facilities. For a bridge deal the focus is the bridge facility; for a development deal it is the development facility being requested.
If there is an aggregate/combined facility that encompasses multiple tranches (e.g. "Structured Mezzanine Facility" = mezz tranche + stretch tranche), pick the aggregate facility as the focus — not the individual tranches.
- focus_facility_id: the id of that facility
- focus_resolution_status: "resolved" | "ambiguous" | "unresolved"
- candidate_facility_ids: ids of every facility considered for focus

## CAPITAL EVENTS
Any repayment/refinance event: { id, facility_id, event_type (created|committed|drawn|partially_repaid|fully_repaid|refinanced|extended|amended|cancelled|maturity), effective_date, trigger, amount, resulting_balance, evidence }.
Every repaid facility MUST have a fully_repaid or refinanced event.

## CAPITAL SNAPSHOTS
A project-level snapshot listing each facility's role: { id, scope_level: "project", stage, as_of_date, positions: [{ facility_id, role }], evidence }.

## VALUATIONS
For each: { id, value_type (market_value|gross_development_value|residual_value|purchase_price|ninety_day_sale_value|day_1_value|unclassified), amount: {amount,currency}, phase_id, valuation_basis, sale_period_days, scenario, evidence }.
- Market value ≠ GDV. Market value is the current asset value; GDV is post-construction gross development value.
- GDV ≠ 90-day sale value (a forced-sale discount).
- Phase-specific GDV ≠ full-scheme GDV — keep as separate valuations with different phase_id.
- A valuation report's "finance rate"/"LTV" is NOT the loan interest rate.

## PHASES
For each: { id, name, sequence, status, unit_count }.

## CLASSIFICATION
- capital_position: "debt" or "equity"
- asset_class_primary: e.g. "residential", "retail", "commercial", "mixed"
- project_type: { category: "development" | "acquisition" | "refinance" | "hold" }
- loan_type: { category: "bridge" | "development" | "investment_term" } — from the focus facility
- loan_rank: { category: "senior" | "mezzanine" | "junior" | "whole_loan" } — from the focus facility role
- operating_model: { category: "income_property" | "development" | "trading" }

## DETAILS (legacy flat projection — derive from the focus facility; NEVER sum unrelated facilities)
Every details field is a sourced-value object { source, source_type, notes, scope, evidence_text, value }. Emit every field; use "value": null where there is no evidence.
- loan_amount.value.amount = the FOCUS facility commitment as a decimal string — NOT a sum across facilities.
- senior_debt / mezzanine_debt / gross_debt = deal-scope debt only — exclude proposed/future development facilities and repaid facilities.
- developer_equity = the equity facility commitment.
- market_value = the current market value (NOT GDV).
- gross_development_value = full-scheme GDV.
- address = { line1, city, postcode, country }.
- gross_internal_area = GIA in sqft.

## KPIs
- loan_to_value.value = loan_amount / market_value as a percentage — LTV is loan/market value, NOT loan/GDV.
- loan_to_gdv.value = loan_amount / gross_development_value as a percentage.
- build_cost_per_area_unit = { amount, currency: "gbp" } build cost per sqft.

Return ONLY valid JSON: { "canonical_deal": {...}, "details": {...}, "kpis": {...}, "classification": {...} }"#;

/// Repair turn: the agent is told its previous output violated invariants
/// and must return the corrected complete document. Tools are available on
/// repair turns only.
pub const REPAIR_PROMPT: &str = r#"Your previous extraction violated one or more invariants. The issues are listed below, each with a code, a JSON path, and a message. You have read access to the source documents through tools — use `search_documents` and `read_page` to verify the correct values before answering, and `list_documents` to see what is available.

RULES:
- Fix ONLY what is wrong. Keep everything that is already correct.
- Do NOT sum unrelated facilities into loan_amount or gross_debt.
- Amounts are exact decimal strings in absolute units ("25750000"), never millions shorthand.
- The focus facility is the NEW money being requested (disposition "requested"), not repaid/existing/proposed debt.
- Return a JSON PATCH object, not the whole document:
  `{"fixes": {"<json.path.from.the.issue>": <corrected value>, ...}}`
- Use exact paths from the issues (e.g. `canonical_deal.facilities[0].commitment.amount`).
  To delete a bad array element, set its path to null.
- Include ONLY the fields you changed; every field not listed is kept as-is.

Return ONLY the JSON patch object."#;

// ─── Two-pass prompts ────────────────────────────────────────────────────────

pub const PASS_A_STRUCTURE: &str = r#"Extract the deal structure from the documents below: classification, facility graph, focus facility, and document annotations.

Return JSON:
{
  "classification": {...},
  "facilities": [...],
  "capital_events": [...],
  "capital_snapshots": [...],
  "focus_facility_id": "<id>",
  "focus_resolution_status": "resolved" | "ambiguous" | "unresolved",
  "candidate_facility_ids": [...],
  "document_annotations": [...]
}

## CLASSIFICATION
- capital_position: "debt" or "equity" — "debt" if the ask is for a loan/facility; "equity" if the reader is invited to subscribe equity shares/units/fund interests (an equity raise, JV, or syndicated investment — even if the underlying vehicle borrows). An equity IM that mentions the project's senior mortgage does NOT make it a debt deal — classify by what the READER is buying.
- asset_class_primary: e.g. "residential", "retail", "commercial", "mixed"
- project_type: { "category": "development" | "acquisition" | "refinance" | "hold" }
- loan_type: { "category": "bridge" | "development" | "investment_term" } — bridge = short-term ≤12mo; development = construction financing; investment_term = stabilised asset
- loan_rank: { "category": "senior" | "mezzanine" | "junior" | "whole_loan" } — "whole_loan" = one facility covering senior+mezzanine
- operating_model: { "category": "income_property" | "development" | "trading" }

## DOCUMENT ANNOTATIONS
For each document: { "name": "<filename>", "role": "<role>" }
Roles: information_memorandum, term_sheet, valuation_report, cash_flow_model, planning_document, brochure, legal_document, other.

## FACILITIES
Extract EVERY facility: senior debt, mezzanine, whole loan, bridge, development loan, investment-term loan, VAT facility, equity, repaid, proposed, planned.

For each:
- id: stable slug (e.g. "bridge_facility", "dev_facility", "shojin_senior_repaid")
- name, instrument_type (debt_facility|ordinary_equity|preferred_equity|shareholder_loan|vendor_finance)
- role (senior_debt|mezzanine_debt|junior_debt|whole_loan|developer_equity|preferred_equity|other)
- scope_level: "deal" | "project" | null
- loan_type: bridge | development | investment_term
- commitment: { amount, currency } — `amount` is an exact DECIMAL STRING in absolute units ("25750000"), NEVER millions shorthand
- term_months, interest_terms: { base_rate, margin, all_in_rate, interest_type }
- status (candidate|active|repaid|cancelled|superseded), disposition (requested|existing|repaid|proposed)
- lender_name, facility_party_ids: [], tranche_ids: []
- evidence: [{ document, page, quote }]

RULES:
1. Include repaid facilities (status "repaid", disposition "repaid") with a matching capital_event.
2. Include proposed/future facilities (disposition "proposed", scope_level "project").
3. Do NOT sum facilities — each is its own entry.
4. Equity is NOT debt — use instrument_type ordinary_equity/preferred_equity.
5. In an equity deal, the reader invests EQUITY — the sponsor's borrowings (senior mortgage, development facility) are context; give them scope_level "project" and do NOT mark them "requested". The focus should be the equity raise itself (preferred_equity/ordinary_equity instrument) — if no equity facility is stated, leave focus_facility_id null rather than pointing at a mortgage the investor isn't taking.

## FOCUS FACILITY
The focus is the NEW money being requested — not repaid/existing/proposed debt.
- For a bridge deal: the bridge facility with disposition "requested".
- For a development deal: the development facility with disposition "requested".
- If exactly one facility has disposition "requested", that is the focus.

## CAPITAL EVENTS
{ id, facility_id, event_type, effective_date, trigger, amount, resulting_balance, evidence }.
Every repaid facility MUST have a fully_repaid or refinanced event.

## CAPITAL SNAPSHOTS
{ id, scope_level: "project", stage, as_of_date, positions: [{ facility_id, role }], evidence }.

Return ONLY valid JSON."#;

pub const PASS_B_DETAILS: &str = r#"Extract valuations, phases, deal details, and KPIs from the documents below.

## CONTEXT — already extracted
Focus facility: {focus_summary}
Classification: {classification_summary}

Return JSON:
{
  "valuations": [...],
  "phases": [...],
  "details": {...},
  "kpis": {...}
}

## VALUATIONS
For each: { id, value_type, amount: {amount, currency}, phase_id, valuation_basis, sale_period_days, scenario, evidence: [{document, page, quote}] }.

value_type: market_value | gross_development_value | residual_value | purchase_price | ninety_day_sale_value | day_1_value | unclassified.

RULES:
- Market value ≠ GDV. Market value = current asset value; GDV = post-construction value.
- Phase-specific GDV ≠ full-scheme GDV — keep as separate valuations.
- `amount` is a DECIMAL STRING ("32500000"), NEVER millions shorthand.

## PHASES
For each: { id, name, sequence, status, unit_count }.

## DETAILS
Every field is a sourced-value object: { "source": "<filename>", "source_type": "extracted|calculated|assumed|unknown", "notes": <string|null>, "scope": "deal|project|null", "evidence_text": <string|null>, "value": <type-specific|null> }.

Emit every field. Use "value": null where the documents give no evidence.

NOTE: These fields are computed deterministically — leave their value as null:
- loan_amount, gross_debt, senior_debt, mezzanine_debt, developer_equity (from facility graph)
- loan_to_value, loan_to_gdv (computed from loan_amount and valuations)

You MUST extract these fields (with value from the documents, NOT null unless truly absent):
- ai_executive_summary: 2-3 sentence summary of the deal in plain English
- deal_type: "debt" or "equity"
- address: { line1, paon, property_name, street_name, city, postcode, country, region, saon, full_address }
- asset_class: the property type (residential, commercial, mixed, etc.)
- construction_costs: { amount, currency } — total construction/build costs
- total_development_cost: { amount, currency } — total development cost EXCLUDING finance costs (the ungeared basis: land + construction + fees + contingency)
- total_investment_cost: { amount, currency } — total investment / geared cost INCLUDING finance costs, fees and rolled-up interest. TIC is always ≥ TDC; if the document labels a figure "total geared costs" or "total costs including finance", it belongs here
- community_infrastructure_levy: { amount, currency } — CIL liability if stated (e.g. "CIL £426,678"); never fold CIL into planning_status text
- sellable_area: { value: <number>, unit: "sqft" } — net sellable/saleable area (NSA) if the source states one; only when GIA/NIA are not stated
- gross_development_value: { amount, currency } — full-scheme GDV
- market_value: { amount, currency } — current market value (NOT GDV)
- purchase_price: { amount, currency }
- gross_internal_area: { value: <number>, unit: "sqft" } — GIA in sqft
- net_internal_area: { value: <number>, unit: "sqft" } — NIA in sqft
- residential_units: array of { property_type, bedrooms, quantity, floor_area: { value, unit }, expected_gdv: { amount, currency } } — expected_gdv is the PER-UNIT expected sale price/GDV when the documents state one (e.g. "expected average unit prices", a sales schedule, or per-type target prices); never multiply by quantity
- commercial_units: FULL tenancy schedule when the documents contain one — for each let unit: unit_identifier, tenant_name, unit_area: { value, unit }, unit_passing_rent: { amount, currency }, unit_passing_rent_per_area_unit, unit_erv, unit_lease_start, unit_lease_expiry, unit_break_date (YYYY-MM-DD), lease_type (e.g. FRI, IRI), epc_rating, rent_review_mechanism, rent_review_frequency, rent_review_next_due. Populate EVERY field the schedule states — tenant, rent and lease dates are the point of the schedule. For unlet/development units emit the unit with the fields that exist only.
- tenants: flat array of every distinct tenant name
- key_tenant: { name, rent_percentage, credit_rating, track_record } — largest tenant by income share
- passing_rent_per_annum, passing_rent_per_area_unit, net_initial_yield, equivalent_yield, reversionary_yield, stabilised_yield, estimated_rental_value — stated asset-level income and yield figures
- wault_to_break, wault_to_expiry — weighted-average unexpired lease term, stated or schedule-derivable
- vacancy_rate, vacant_area, vacant_possession — stated vacancy metrics
- lease_length_remaining, tenure_length, rent_review_mechanism, rent_review_frequency, rent_review_next_due, tenant_credit_rating — asset-level lease/tenure facts
- net_operating_income, ground_rent_per_annum, operating_costs — stated income/cost figures
- year_built, last_refurbished
- debt_structure: "whole_loan" | "senior_mezzanine" | "senior_only" | null
- interest_rate, loan_term, fees, guarantees, covenants
- use_of_funds, source_of_funds, exit_plan — the ASSET-LEVEL exit: how the investment ends (unit sales / disposal of the completed scheme / refinance into term debt). When the documents describe both a facility-repayment step (e.g. "bridge repaid by the development facility's first advance") and an asset-level exit (e.g. "sale of completed units"), report the asset-level exit — the facility step is a capital event, not the exit plan.
- borrower_details: { name, type, track_record, email, phone }
- sponsor_principals: [{ name, role, company, track_record: [{ name, description }] }] — extract EVERY named principal/director/shareholder from developer CVs, bios and structure charts; include a per-person track_record entry for each stated project or role.
- planning_status, epc_rating, tenure
- construction_timelines: { start_date, completion_date, duration_months }
- insurance_requirements, capex_plan
- current_asset_description: description of the property in its current state
- building_specification: description of planned/completed works

## DEAL-TYPE FIELD EMPHASIS
Use the classification from pass A to prioritise:

- Acquisition / bridge on an income-producing asset (retail park, office, PLR-style): extract asking_price, purchase_price, day_1_value, market_value, purchaser_costs (SDLT + agent + legal — may be a money amount OR a percentage of price; extract whichever is stated into purchaser_costs), the FULL commercial_units tenancy schedule, passing_rent_per_annum, net_initial_yield, wault_to_break, wault_to_expiry, vacancy_rate, key_tenant, rent_review_next_due, plus loan_term, interest_rate, fees, exit_plan, lender_tenant_exposure.
- Development deal: project_gdv AND gross_development_value (full scheme), total_development_cost, total_investment_cost, construction_costs, residential_units with per-unit expected_gdv, phases, construction_timelines, planning_status, profit stated figures.
- Equity deal: deal_type "equity", purchase_price, purchaser_costs, the equity raise amount, target IRR/return metrics stated to investors, and the underlying asset economics (rent, value, NIY) — equity docs still describe the asset.
- Refinance / disposal bridge: market_value, day_1_value, stabilised_value, loan_term, interest_rate, fees, exit_plan, current_asset_description.

## KPIs
Every KPI is: { source, source_type, notes, scope, evidence_text, value }.

Extract these KPIs from the documents if explicitly stated:
- build_cost_per_area_unit: { amount, currency } — cost per sqft
- profit_on_cost: percentage — (GDV - TDC) / TDC * 100
- interest_coverage_ratio, debt_yield, net_initial_yield, gross_yield
- development_margin, sales_rate_per_sqft
- Any other KPI mentioned in the documents.

Set value to null for KPIs not stated or not computable from the documents.

Return ONLY valid JSON."#;
