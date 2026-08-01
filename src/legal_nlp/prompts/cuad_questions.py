from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class CUADQuestion:
    """
    Represents one official CUAD Question Answering task.

    Each question corresponds to one clause category in the
    Contract Understanding Atticus Dataset (CUAD).
    """

    clause_type: str
    prompt: str


CUAD_QUESTIONS: List[CUADQuestion] = [

    # ==========================================================
    # Basic Contract Metadata
    # ==========================================================

    CUADQuestion(
        clause_type="document_name",
        prompt='Highlight the parts (if any) related to "Document Name".'
    ),

    CUADQuestion(
        clause_type="parties",
        prompt='Highlight the parts (if any) related to "Parties".'
    ),

    CUADQuestion(
        clause_type="agreement_date",
        prompt='Highlight the parts (if any) related to "Agreement Date".'
    ),

    CUADQuestion(
        clause_type="effective_date",
        prompt='Highlight the parts (if any) related to "Effective Date".'
    ),

    # ==========================================================
    # Term & Termination
    # ==========================================================

    CUADQuestion(
        clause_type="expiration_date",
        prompt='Highlight the parts (if any) related to "Expiration Date".'
    ),

    CUADQuestion(
        clause_type="renewal_term",
        prompt='Highlight the parts (if any) related to "Renewal Term".'
    ),

    CUADQuestion(
        clause_type="notice_to_terminate_renewal",
        prompt='Highlight the parts (if any) related to "Notice to Terminate Renewal".'
    ),

    CUADQuestion(
        clause_type="notice_period_to_terminate_for_convenience",
        prompt='Highlight the parts (if any) related to "Notice Period to Terminate for Convenience".'
    ),

    CUADQuestion(
        clause_type="termination_for_convenience",
        prompt='Highlight the parts (if any) related to "Termination for Convenience".'
    ),

    CUADQuestion(
        clause_type="rofr_rofo_rofn",
        prompt='Highlight the parts (if any) related to "Right of First Refusal, First Offer or First Negotiation".'
    ),

    # ==========================================================
    # Restrictive Covenants
    # ==========================================================

    CUADQuestion(
        clause_type="non_compete",
        prompt='Highlight the parts (if any) related to "Non-Compete".'
    ),

    CUADQuestion(
        clause_type="exclusivity",
        prompt='Highlight the parts (if any) related to "Exclusivity".'
    ),

    CUADQuestion(
        clause_type="no_solicit_customers",
        prompt='Highlight the parts (if any) related to "No-Solicit of Customers".'
    ),

    CUADQuestion(
        clause_type="no_solicit_employees",
        prompt='Highlight the parts (if any) related to "No-Solicit of Employees".'
    ),

    CUADQuestion(
        clause_type="non_disparagement",
        prompt='Highlight the parts (if any) related to "Non-Disparagement".'
    ),

    # ==========================================================
    # IP & Liability
    # ==========================================================

    CUADQuestion(
        clause_type="anti_assignment",
        prompt='Highlight the parts (if any) related to "Anti-Assignment".'
    ),

    CUADQuestion(
        clause_type="change_of_control",
        prompt='Highlight the parts (if any) related to "Change of Control".'
    ),

    CUADQuestion(
        clause_type="most_favored_nation",
        prompt='Highlight the parts (if any) related to "Most Favored Nation".'
    ),

    CUADQuestion(
        clause_type="uncapped_liability",
        prompt='Highlight the parts (if any) related to "Uncapped Liability".'
    ),

    CUADQuestion(
        clause_type="cap_on_liability",
        prompt='Highlight the parts (if any) related to "Cap on Liability".'
    ),

    CUADQuestion(
        clause_type="liquidated_damages",
        prompt='Highlight the parts (if any) related to "Liquidated Damages".'
    ),

    CUADQuestion(
        clause_type="warranty_duration",
        prompt='Highlight the parts (if any) related to "Warranty Duration".'
    ),

    CUADQuestion(
        clause_type="insurance",
        prompt='Highlight the parts (if any) related to "Insurance".'
    ),

    CUADQuestion(
        clause_type="covenant_not_to_sue",
        prompt='Highlight the parts (if any) related to "Covenant Not to Sue".'
    ),

    CUADQuestion(
        clause_type="third_party_beneficiary",
        prompt='Highlight the parts (if any) related to "Third Party Beneficiary".'
    ),

    CUADQuestion(
        clause_type="ip_ownership_assignment",
        prompt='Highlight the parts (if any) related to "IP Ownership Assignment".'
    ),

    CUADQuestion(
        clause_type="joint_ip_ownership",
        prompt='Highlight the parts (if any) related to "Joint IP Ownership".'
    ),

    # ==========================================================
    # Financial & Audit
    # ==========================================================

    CUADQuestion(
        clause_type="audit_rights",
        prompt='Highlight the parts (if any) related to "Audit Rights".'
    ),

    CUADQuestion(
        clause_type="revenue_profit_sharing",
        prompt='Highlight the parts (if any) related to "Revenue/Profit Sharing".'
    ),

    CUADQuestion(
        clause_type="price_restriction",
        prompt='Highlight the parts (if any) related to "Price Restriction".'
    ),

    CUADQuestion(
        clause_type="volume_restriction",
        prompt='Highlight the parts (if any) related to "Volume Restriction".'
    ),

    CUADQuestion(
        clause_type="minimum_commitment",
        prompt='Highlight the parts (if any) related to "Minimum Commitment".'
    ),

    # ==========================================================
    # General / Boilerplate
    # ==========================================================

    CUADQuestion(
        clause_type="governing_law",
        prompt='Highlight the parts (if any) related to "Governing Law".'
    ),

    CUADQuestion(
        clause_type="venue",
        prompt='Highlight the parts (if any) related to "Venue".'
    ),

    CUADQuestion(
        clause_type="license_grant",
        prompt='Highlight the parts (if any) related to "License Grant".'
    ),

    CUADQuestion(
        clause_type="post_termination_services",
        prompt='Highlight the parts (if any) related to "Post-Termination Services".'
    ),

    CUADQuestion(
        clause_type="survival",
        prompt='Highlight the parts (if any) related to "Survival".'
    ),

    CUADQuestion(
        clause_type="cap_on_liability_exceptions",
        prompt='Highlight the parts (if any) related to "Cap on Liability Exceptions".'
    ),

    CUADQuestion(
        clause_type="indemnification",
        prompt='Highlight the parts (if any) related to "Indemnification".'
    ),

    CUADQuestion(
        clause_type="affiliate_license_licensee",
        prompt='Highlight the parts (if any) related to "Affiliate License - Licensee".'
    ),

    CUADQuestion(
        clause_type="affiliate_license_licensor",
        prompt='Highlight the parts (if any) related to "Affiliate License - Licensor".'
    ),
]