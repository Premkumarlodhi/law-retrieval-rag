"""
graph_schema.py

Legal Knowledge Graph ontology.

This schema constrains the LLMGraphTransformer to generate a consistent
knowledge graph for legal contracts.
"""

# =============================================================================
# Node Labels
# =============================================================================

CORE_NODES = [
    "Contract",
    "Clause",
    "Section",
]

PARTY_NODES = [
    "Company",
    "Party",
    "Person",
]

LEGAL_NODES = [
    "Law",
    "Jurisdiction",
    "Regulation",
]

COMMERCIAL_NODES = [
    "Product",
    "Service",
    "Payment",
    "Currency",
]

RIGHTS_AND_OBLIGATIONS = [
    "Obligation",
    "Right",
    "Restriction",
]

TEMPORAL_NODES = [
    "Date",
    "Duration",
    "Deadline",
]

EVENT_NODES = [
    "TerminationEvent",
    "ForceMajeureEvent",
    "BreachEvent",
]

IP_NODES = [
    "IP",
    "Patent",
    "Trademark",
    "Copyright",
]

MISC_NODES = [
    "Location",
    "Document",
]

ALLOWED_NODES = (
    CORE_NODES
    + PARTY_NODES
    + LEGAL_NODES
    + COMMERCIAL_NODES
    + RIGHTS_AND_OBLIGATIONS
    + TEMPORAL_NODES
    + EVENT_NODES
    + IP_NODES
    + MISC_NODES
)

# =============================================================================
# Relationship Types
# =============================================================================

STRUCTURE_RELATIONSHIPS = [
    "HAS_PARTY",
    "HAS_CLAUSE",
    "HAS_SECTION",
    "HAS_SCHEDULE",
    "HAS_APPENDIX",
    "HAS_EXHIBIT",
]

PARTICIPATION_RELATIONSHIPS = [
    "PARTY_TO",
    "SIGNS",
    "SIGNED_BY",
]

LEGAL_RELATIONSHIPS = [
    "GOVERNED_BY",
    "REGULATED_BY",
    "REFERENCES",
    "AMENDS",
    "SUPERSEDES",
]

CLAUSE_RELATIONSHIPS = [
    "CONTAINS",
    "MENTIONS",
    "DEFINES",
    "EXCLUDES",
]

OBLIGATION_RELATIONSHIPS = [
    "HAS_OBLIGATION",
    "OWES",
    "PAYS",
    "RECEIVES",
    "MUST",
    "SHALL",
    "MAY",
    "PROHIBITS",
]

COMMERCIAL_RELATIONSHIPS = [
    "SUPPLIES",
    "PURCHASES",
    "SELLS",
    "DELIVERS",
    "USES",
]

IP_RELATIONSHIPS = [
    "OWNS",
    "LICENSES",
    "TRANSFERS",
    "PROTECTS",
]

PAYMENT_RELATIONSHIPS = [
    "HAS_PAYMENT",
    "PAYABLE_IN",
    "HAS_PRICE",
]

TIME_RELATIONSHIPS = [
    "STARTS_ON",
    "ENDS_ON",
    "TERMINATES_ON",
    "HAS_DURATION",
    "HAS_DEADLINE",
]

EVENT_RELATIONSHIPS = [
    "TERMINATED_BY",
    "TRIGGERED_BY",
    "CAUSES",
    "RESULTS_IN",
]

LOCATION_RELATIONSHIPS = [
    "LOCATED_IN",
    "EXECUTED_IN",
]

MISC_RELATIONSHIPS = [
    "RELATED_TO",
]

ALLOWED_RELATIONSHIPS = (
    STRUCTURE_RELATIONSHIPS
    + PARTICIPATION_RELATIONSHIPS
    + LEGAL_RELATIONSHIPS
    + CLAUSE_RELATIONSHIPS
    + OBLIGATION_RELATIONSHIPS
    + COMMERCIAL_RELATIONSHIPS
    + IP_RELATIONSHIPS
    + PAYMENT_RELATIONSHIPS
    + TIME_RELATIONSHIPS
    + EVENT_RELATIONSHIPS
    + LOCATION_RELATIONSHIPS
    + MISC_RELATIONSHIPS
)

# =============================================================================
# Properties
# =============================================================================

NODE_PROPERTIES = [
    "name",
    "type",
    "description",
    "contract_type",
    "title",
    "section",
    "doc_id",
    "chunk_id",
    "source",
    "confidence",
]

RELATIONSHIP_PROPERTIES = [
    "description",
    "source",
    "confidence",
    "evidence",
]

# =============================================================================
# Extraction Schema
#
# This is intentionally smaller than the master ontology.
# The LLMGraphTransformer performs better with a constrained vocabulary.
# =============================================================================

GRAPH_EXTRACTION_NODES = [
    "Contract",
    "Company",
    "Party",
    "Clause",
    "Section",
    "Jurisdiction",
    "Law",
    "Obligation",
    "Payment",
    "Date",
    "TerminationEvent",
    "ForceMajeureEvent",
]

GRAPH_EXTRACTION_RELATIONSHIPS = [
    "HAS_PARTY",
    "PARTY_TO",
    "HAS_CLAUSE",
    "HAS_SECTION",
    "HAS_OBLIGATION",
    "GOVERNED_BY",
    "REFERENCES",
    "MENTIONS",
    "OWES",
    "PAYS",
    "TERMINATED_BY",
    "HAS_PAYMENT",
    "STARTS_ON",
    "ENDS_ON",
]

# =============================================================================
# Helper
# =============================================================================

def get_transformer_kwargs():
    """
    Returns keyword arguments for LLMGraphTransformer.
    """

    return {
        "allowed_nodes": GRAPH_EXTRACTION_NODES,
        "allowed_relationships": GRAPH_EXTRACTION_RELATIONSHIPS,
        "node_properties": NODE_PROPERTIES,
        "relationship_properties": RELATIONSHIP_PROPERTIES,
        "strict_mode": True,
    }