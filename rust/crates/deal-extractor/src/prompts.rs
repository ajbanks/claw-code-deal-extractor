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
2. Include future/planned facilities (e.g. a development facility that will replace the bridge) with disposition "proposed" and scope_level "project". ALWAYS give a proposed facility its commitment when the pack states or models it (e.g. the appraisal's development facility line, "gross development facility £35.8m", or a facility sized off TDC) — a proposed facility without an amount hides half the ask. Only omit commitment when no figure exists anywhere.
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
- loan_balance_including_interest = { amount, currency } — the stated all-in loan figure (e.g. "loan incl. interest £5,699,659", a redemption or maturity balance) when the pack quotes one; put the quoted sentence in evidence_text.
- finance_costs (details) = { amount, currency } — a stated total interest/finance cost over the term (e.g. "Finance costs £749,659"). Stated amount only — never rate × term.

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

/// Recall fill: gaps the extraction left — evidenced-but-empty fields and
/// corpus figures that appear nowhere in the output. Patch may only ADD.
pub const RECALL_FILL_PROMPT: &str = r#"An extraction of this deal is below, followed by GAPS: fields that are empty although the documents appear to answer them, and figures stated in the documents that appear nowhere in the extraction. You have read access to the documents through `read_page`, `search_documents` and `list_documents` — read the cited page before deciding.

For each gap:
- If it is a fact about THIS deal (loan/facility, value, GDV, price, cost, rent, yield, fee, term, area, unit schedule, party), add it at the correct contract path.
- If it is NOT a deal fact (a comparable, a sensitivity/illustration scenario, a market statistic, a monthly cash-flow line, a figure already captured under a different rounding or as part of a total), skip it.
- Never overwrite a populated field — only empty fields or new array entries (append at index = current array length) are accepted.

SHAPE RULES:
- `details.<field>` → a WithSource object: {"value": <value>, "source": "<document name>", "source_type": "extracted", "evidence_text": "<short quote>", "notes": null, "scope": "deal"|"project"|null}
- `kpis.<field>` → the same WithSource shape.
- `canonical_deal.<array>[<i>]` → a raw entry matching the shape of existing entries (e.g. valuations: {id, value_type, amount: {amount, currency}, phase_id, valuation_basis, evidence: [{document, page, quote}]}).
- Money values are {amount, currency}; canonical_deal amounts are exact decimal strings in absolute units ("25750000"), never millions shorthand.

Return `{"fixes": {"<path>": <value>, ...}}` with ONLY the paths you are adding. Return `{"fixes": {}}` if every gap is irrelevant.

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
2. Include proposed/future facilities (disposition "proposed", scope_level "project") — ALWAYS with their commitment when the pack states or models it; a proposed facility without an amount hides half the ask.
3. Do NOT sum facilities — each is its own entry.
4. Equity is NOT debt — use instrument_type ordinary_equity/preferred_equity.
5. In an equity deal, the reader invests EQUITY — the sponsor's borrowings (senior mortgage, development facility) are context; give them scope_level "project" and do NOT mark them "requested". The focus should be the equity raise itself (preferred_equity/ordinary_equity instrument) — if no equity facility is stated, leave focus_facility_id null rather than pointing at a mortgage the investor isn't taking.

## FOCUS FACILITY
The focus is the NEW money being requested — not repaid/existing/proposed debt.
- For a bridge deal: the bridge facility with disposition "requested".
- For a development deal: the development facility with disposition "requested".
- If exactly one facility has disposition "requested", that is the focus.
- If NO facility is being requested (a structured transaction such as sale & leaseback, forward purchase, promotion agreement, or an extension/variation of existing debt), leave focus_facility_id null — do not point it at an existing facility.
- A maturity extension or amendment asks to vary an EXISTING facility, not to take new money — mark that facility disposition "existing" and focus_facility_id null.

## CAPITAL EVENTS
{ id, facility_id, event_type, effective_date, trigger, amount, resulting_balance, evidence }.
Every repaid facility MUST have a fully_repaid or refinanced event.

## CAPITAL SNAPSHOTS
{ id, scope_level: "project", stage, as_of_date, positions: [{ facility_id, role }], evidence }.

Return ONLY valid JSON."#;

/// Pass-A adjudication: two independent structure extractions disagreed on
/// specific points; the model decides each from the evidence and returns one
/// complete structure JSON in PASS A's format.
pub const PASS_A_ADJUDICATE: &str = r#"Two independent extractions of this deal's structure disagree on the points listed below. Read the documents, decide each point from the evidence, and return the single correct complete structure JSON in exactly the same format as the inputs (same keys as PASS A). Keep everything the two agree on. Include a facility only if the documents support it."#;

/// Shared prelude for the three pass-B sub-passes: the already-extracted
/// structure context and the sourced-value object rules.
pub const PASS_B_HEADER: &str = r#"Extract deal data from the documents below.

## CONTEXT — already extracted
Focus facility: {focus_summary}
Classification: {classification_summary}

Every details field is a sourced-value object: { "source": "<filename>", "source_type": "extracted|calculated|assumed|unknown", "notes": <string|null>, "scope": "deal|project|null", "evidence_text": <string|null>, "value": <type-specific|null>, "confidence": <0.0-1.0 or omit> }. Include "confidence" on monetary/quantity values: ~1.0 when stated explicitly in the source, lower when inferred or ambiguous — this lets downstream arbitration catch wrong-but-plausible figures.

Emit every field. Use "value": null where the documents give no evidence.

NOTE: These fields are computed deterministically — leave their value as null:
- loan_amount, gross_debt, senior_debt, mezzanine_debt, developer_equity (from facility graph)
- loan_to_value, loan_to_gdv (computed from loan_amount and valuations)
"#;

/// Pass B sub-pass 1: valuations, phases, economics details, and KPIs.
pub const PASS_B_ECONOMICS: &str = r#"## VALUATIONS
For each: { id, value_type, amount: {amount, currency}, phase_id, valuation_basis, sale_period_days, scenario, evidence: [{document, page, quote}] }.

value_type: market_value | gross_development_value | residual_value | purchase_price | ninety_day_sale_value | day_1_value | unclassified.

RULES:
- Market value ≠ GDV. Market value = current asset value; GDV = post-construction value.
- Phase-specific GDV ≠ full-scheme GDV — keep as separate valuations.
- `amount` is a DECIMAL STRING ("32500000"), NEVER millions shorthand.

## PHASES
For each: { id, name, sequence, status, unit_count }.

## DETAILS
You MUST extract these fields (with value from the documents, NOT null unless truly absent):
- deal_type: "debt" or "equity"
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
- debt_structure: "whole_loan" | "senior_mezzanine" | "senior_only" | null
- interest_rate, loan_term, fees, guarantees, covenants
- use_of_funds, source_of_funds, exit_plan — the ASSET-LEVEL exit: how the investment ends (unit sales / disposal of the completed scheme / refinance into term debt). When the documents describe both a facility-repayment step (e.g. "bridge repaid by the development facility's first advance") and an asset-level exit (e.g. "sale of completed units"), report the asset-level exit — the facility step is a capital event, not the exit plan.
- finance_costs, purchaser_costs, project_gdv, day_1_value, stabilised_value — every other stated money/cost/value figure

## DEAL-TYPE FIELD EMPHASIS
{deal_type_guidance}

## KPIs
Every KPI is: { source, source_type, notes, scope, evidence_text, value }.

Extract these KPIs from the documents if explicitly stated:
- build_cost_per_area_unit: { amount, currency } — cost per sqft
- profit_on_cost: percentage — (GDV - TDC) / TDC * 100
- interest_coverage_ratio, debt_yield, net_initial_yield, gross_yield
- development_margin, sales_rate_per_sqft
- Any other KPI mentioned in the documents.

Set value to null for KPIs not stated or not computable from the documents."#;

/// Pass B sub-pass 2: the income side — tenancy schedule, rents, yields,
/// lease metrics. Returns {"details": {...}} only.
pub const PASS_B_INCOME: &str = r#"## DETAILS
You MUST extract these fields (with value from the documents, NOT null unless truly absent):
- commercial_units: FULL tenancy schedule when the documents contain one — for each let unit: unit_identifier, tenant_name, unit_area: { value, unit }, unit_passing_rent: { amount, currency }, unit_passing_rent_per_area_unit, unit_erv, unit_lease_start, unit_lease_expiry, unit_break_date (YYYY-MM-DD), unit_break_penalty { amount, currency }, unit_break_notice_months, unit_extension_approach, unit_lease_notes (rent-free, exclusivity, other stated per-unit terms), lease_type (e.g. FRI, IRI), epc_rating, rent_review_mechanism, rent_review_frequency, rent_review_next_due. Populate EVERY field the schedule states — tenant, rent and lease dates are the point of the schedule. For unlet/development units emit the unit with the fields that exist only.
- tenants: flat array of every distinct tenant name
- key_tenant: { name, rent_percentage, credit_rating, track_record } — largest tenant by income share
- passing_rent_per_annum, passing_rent_per_area_unit, net_initial_yield, equivalent_yield, reversionary_yield, stabilised_yield, estimated_rental_value — stated asset-level income and yield figures
- wault_to_break, wault_to_expiry — weighted-average unexpired lease term, stated or schedule-derivable
- vacancy_rate, vacant_area, vacant_possession — stated vacancy metrics
- lease_length_remaining, tenure_length, rent_review_mechanism, rent_review_frequency, rent_review_next_due, tenant_credit_rating — asset-level lease/tenure facts
- net_operating_income, ground_rent_per_annum, operating_costs — stated income/cost figures
- epc_rating, tenure
- year_built, last_refurbished"#;

/// Pass B sub-pass 3: parties, narrative and descriptive fields.
/// Returns {"details": {...}} only.
pub const PASS_B_PARTIES: &str = r#"## DETAILS
You MUST extract these fields (with value from the documents, NOT null unless truly absent):
- ai_executive_summary: 2-3 sentence summary of the deal in plain English
- address: { line1, paon, property_name, street_name, city, postcode, country, region, saon, full_address }
- agent_details: { name, email, phone } — the selling/marketing agent or named contact in the pack's contact block (e.g. "Morgan Williams", contact footers). Always extract when stated.
- document_date: the pack/publication date the document states (e.g. "April 2024" particulars, a dated report, an "as at" date). Preserve the stated precision — month-year when no day is given.
- borrower_details: { name, type, track_record, email, phone }
- sponsor_principals: [{ name, role, company, track_record: [{ name, description }] }] — extract EVERY named principal/director/shareholder from developer CVs, bios and structure charts; include a per-person track_record entry for each stated project or role.
- planning_status
- construction_timelines: { start_date, completion_date, duration_months }
- insurance_requirements, capex_plan
- current_asset_description: description of the property in its current state
- building_specification: description of planned/completed works"#;

// ─── Per-deal-type emphasis blocks ───────────────────────────────────────────
// PASS_B is the details pass — its field emphasis should only carry guidance
// for the deal type in front of it. An always-on list of every archetype
// bloats the prompt and dilutes the signal; pass A's classification chooses
// which blocks apply.

use serde_json::Value;

const G_ACQ_INCOME: &str = "- Acquisition / bridge on an income-producing asset (retail park, office, PLR-style): asking_price, purchase_price, day_1_value, market_value, purchaser_costs (SDLT + agent + legal — money amount OR % of price), the FULL commercial_units tenancy schedule, passing_rent_per_annum, net_initial_yield, wault_to_break, wault_to_expiry, vacancy_rate, key_tenant, rent_review_next_due, loan_term, interest_rate, fees, exit_plan, lender_tenant_exposure.";
const G_DEVELOPMENT: &str = "- Development deal: project_gdv AND gross_development_value (full scheme), total_development_cost, total_investment_cost, construction_costs, residential_units with per-unit expected_gdv AND expected_rent_pcm, phases, construction_timelines, planning_status, profit stated figures.";
const G_EQUITY: &str = "- Equity deal: deal_type \"equity\", purchase_price, purchaser_costs, the equity raise amount, target IRR/return metrics stated to investors, and the underlying asset economics (rent, value, NIY) — equity docs still describe the asset.";
const G_REFI_BRIDGE: &str = "- Refinance / disposal bridge: market_value, day_1_value, stabilised_value, loan_term, interest_rate, fees, exit_plan, current_asset_description.";
const G_STRUCTURED: &str = "- Structured non-debt transactions (sale & leaseback, ground lease, income strip, forward funding / forward commitment, promotion agreement): DO NOT fabricate a loan — the pack describes an investor paying a capital sum or rent stream, not a lender advancing a facility. Emit the facilities that actually exist (often none). Extract deal_type, finance_structure, sale_price / strip_price, leaseback_rent_per_annum / strip_rent_per_annum, leaseback_term_years / strip_term_years, rent_review, buyback_option, reversion, forward_purchase_price, forward_commitment, promoter_share.";
const G_PLANNING: &str = "- Planning-gain / strategic land (option agreement, promotion agreement, hybrid): current market_value vs consented_value are distinct — do not merge them. Extract option_agreement, promotion_agreement, planning_budget, overage_clause, deferred_consideration.";
const G_ISLAMIC: &str = "- Islamic finance (diminishing musharaka, ijara): the financier's contribution is the facility (debt_facility, role senior_debt); the customer_contribution is equity. Extract finance_structure, customer_contribution.";
const G_RP: &str = "- Social housing / registered provider: the lending basis is euv_sh_value; open_market_value_vacant_possession is an information-only comparator — extract both separately.";
const G_SPECIALIST_RESI: &str = "- Specialist residential (HMO, MUFB, PBSA): extract rental_assessment (e.g. gross rent less 25% deduction) and nomination_agreement (operator, beds, coverage).";
const G_TRADING: &str = "- Trading assets (care home, hotel, nursery, pub): adjusted_ebitda is the sizing figure; market_value may carry an MV1/MV2/MV3 basis note.";
const G_WORKOUT: &str = "- Workouts / amendments (maturity extension, covenant breach, cure, stressed refi): the existing facility usually IS the answer — extract existing_debt, extension_request, covenant_status, amendment_fee, warrants as stated.";
const G_SECOND_CHARGE: &str = "- Second charge / additional debt: existing_debt stays in place; the ask is incremental. Note combined-LTV context in notes.";
const G_PORTFOLIO: &str = "- Portfolio / cross-collateralised: extract portfolio_assets with per-asset value and release price; market_value is the aggregate.";
const G_GENERIC: &str = "- Extract every stated valuation, cost, income and structure field in the pack — purchase/market values, GDV, costs, rents/yields, loan terms, structure terms — per the field list above.";

/// Build the `{deal_type_guidance}` block for PASS_B from pass A's
/// classification. Only blocks matching the deal's signals are included —
/// a bridge pack shouldn't see specialist-resi or Islamic-finance prose.
/// Falls back to the generic line when classification gives no signal.
pub fn deal_type_guidance(classification: &Value) -> String {
    let text = serde_json::to_string(classification)
        .unwrap_or_default()
        .to_lowercase();
    let cat = |key: &str| {
        classification
            .get(key)
            .and_then(|o| o.get("category"))
            .and_then(|c| c.as_str())
            .unwrap_or("")
    };
    let has = |re_terms: &[&str]| re_terms.iter().any(|t| text.contains(t));

    let capital = classification
        .get("capital_position")
        .and_then(Value::as_str)
        .unwrap_or("");
    let project = cat("project_type");
    let loan = cat("loan_type");
    let operating = cat("operating_model");

    let mut blocks: Vec<&'static str> = Vec::new();
    let mut push = |b: &'static str| {
        if !blocks.contains(&b) {
            blocks.push(b)
        }
    };

    // Keyword hits on the serialised classification (descriptions, signals)
    // catch the archetypes that have no enum category of their own.
    if has(&[
        "leaseback",
        "income strip",
        "income_strip",
        "forward fund",
        "forward_fund",
        "forward commitment",
        "promotion agreement",
        "ground rent",
    ]) {
        push(G_STRUCTURED);
    }
    if has(&[
        "option agreement",
        "promotion agreement",
        "planning gain",
        "hope value",
        "overage",
        "strategic land",
    ]) {
        push(G_PLANNING);
    }
    if has(&["musharaka", "ijara", "islamic", "sharia"]) {
        push(G_ISLAMIC);
    }
    if has(&["registered provider", "social housing", "euv", "euv-sh"]) {
        push(G_RP);
    }
    if has(&[
        "hmo",
        "mufb",
        "pbsa",
        "student accommodation",
        "student housing",
    ]) {
        push(G_SPECIALIST_RESI);
    }
    if has(&[
        "covenant breach",
        "maturity extension",
        "amendment",
        "workout",
        "receivership",
        "cure",
    ]) {
        push(G_WORKOUT);
    }
    if has(&["second charge", "second_charge", "additional debt"]) {
        push(G_SECOND_CHARGE);
    }
    if has(&["portfolio", "cross-collateral", "cross collateral"]) {
        push(G_PORTFOLIO);
    }

    // Category-driven matches.
    if capital == "equity" || has(&["equity raise", "subscribe", "share capital"]) {
        push(G_EQUITY);
    }
    if project == "development"
        || operating == "development"
        || loan == "development"
        || has(&["gdv", "construction", "build programme"])
    {
        push(G_DEVELOPMENT);
    }
    if loan == "bridge" || project == "refinance" || has(&["bridge", "refinanc", "disposal"]) {
        push(G_REFI_BRIDGE);
    }
    if loan == "investment_term" || operating == "income_property" || project == "acquisition" {
        push(G_ACQ_INCOME);
    }
    if operating == "trading"
        || has(&[
            "ebitda",
            "care home",
            "hotel",
            "nursery",
            "pub",
            "holiday park",
        ])
    {
        push(G_TRADING);
    }

    if blocks.is_empty() {
        blocks.push(G_GENERIC);
    }
    blocks.join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;

    /// Field names appearing in the `- name[, name…]` bullets of a pass-B
    /// section's `## DETAILS` block, keyed by the pass that owns them.
    /// (KPI and valuation bullets elsewhere in the const are not details.)
    fn bullet_fields(section: &str) -> Vec<String> {
        let mut out = Vec::new();
        let mut in_details = false;
        for line in section.lines() {
            let line = line.trim();
            if line.starts_with("## ") {
                in_details = line.contains("DETAILS");
                continue;
            }
            if !in_details {
                continue;
            }
            let Some(rest) = line.strip_prefix("- ") else {
                continue;
            };
            // Field names run up to the first ':' or ' —' description.
            let head = rest.split(':').next().unwrap_or(rest);
            let head = head.split(" — ").next().unwrap_or(head);
            for tok in head.split([',', '/']).map(str::trim) {
                if !tok.is_empty()
                    && tok
                        .chars()
                        .all(|c| c.is_ascii_lowercase() || c == '_' || c.is_ascii_digit())
                {
                    out.push(tok.to_string());
                }
            }
        }
        out
    }

    /// Every details field the old single-pass PASS_B listed, copied from the
    /// pre-split prompt. Each must land in exactly one of the three sub-passes.
    const OLD_PASS_B_FIELDS: &[&str] = &[
        "ai_executive_summary",
        "deal_type",
        "address",
        "asset_class",
        "construction_costs",
        "total_development_cost",
        "total_investment_cost",
        "community_infrastructure_levy",
        "sellable_area",
        "gross_development_value",
        "market_value",
        "purchase_price",
        "gross_internal_area",
        "net_internal_area",
        "residential_units",
        "commercial_units",
        "agent_details",
        "document_date",
        "tenants",
        "key_tenant",
        "passing_rent_per_annum",
        "passing_rent_per_area_unit",
        "net_initial_yield",
        "equivalent_yield",
        "reversionary_yield",
        "stabilised_yield",
        "estimated_rental_value",
        "wault_to_break",
        "wault_to_expiry",
        "vacancy_rate",
        "vacant_area",
        "vacant_possession",
        "lease_length_remaining",
        "tenure_length",
        "rent_review_mechanism",
        "rent_review_frequency",
        "rent_review_next_due",
        "tenant_credit_rating",
        "net_operating_income",
        "ground_rent_per_annum",
        "operating_costs",
        "year_built",
        "last_refurbished",
        "debt_structure",
        "interest_rate",
        "loan_term",
        "fees",
        "guarantees",
        "covenants",
        "use_of_funds",
        "source_of_funds",
        "exit_plan",
        "borrower_details",
        "sponsor_principals",
        "planning_status",
        "epc_rating",
        "tenure",
        "construction_timelines",
        "insurance_requirements",
        "capex_plan",
        "current_asset_description",
        "building_specification",
    ];

    #[test]
    fn pass_b_split_covers_every_old_field_exactly_once() {
        let mut seen: BTreeMap<String, usize> = BTreeMap::new();
        for section in [PASS_B_ECONOMICS, PASS_B_INCOME, PASS_B_PARTIES] {
            for f in bullet_fields(section) {
                *seen.entry(f).or_default() += 1;
            }
        }
        let dupes: Vec<&String> = seen
            .iter()
            .filter(|(_, n)| **n > 1)
            .map(|(f, _)| f)
            .collect();
        assert!(dupes.is_empty(), "fields in multiple passes: {dupes:?}");
        for field in OLD_PASS_B_FIELDS {
            assert!(
                seen.contains_key(*field),
                "{field} not covered by any pass-B sub-pass"
            );
        }
    }
}
