"""Manually curated Checkpoint 11 evaluation dataset.

Every supported fact and evidence locator below comes directly from one of the five
Atlas policy documents.
The retrieval queries are fixed before provider comparison so embeddings, chunking,
and reranking receive identical information needs.
"""

from __future__ import annotations

from typing import Any


def evidence(source: str, contains: str) -> dict[str, str]:
    return {"source": source, "contains": contains}


def queries(original: str, first: str, second: str, third: str) -> list[str]:
    return [original, first, second, third]


def case(
    case_id: str,
    category: str,
    question: str,
    expected_evidence: list[dict[str, str]],
    expected_facts: list[list[str]],
    search_queries: list[str],
    *,
    history: list[dict[str, str]] | None = None,
    authorization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "id": case_id,
        "category": category,
        "question": question,
        "history": history or [],
        "expected_evidence": expected_evidence,
        "expected_facts": expected_facts,
        "search_queries": search_queries,
        "authorization": authorization or {
            "tenant_id": "atlas-logistics",
            "access_scopes": ["employees"],
        },
    }


SLA = "shipment_sla_policy.md"
DISPATCH = "dispatch_operations_sop.md"
EXCEPTION = "delivery_exception_policy.md"
CUSTOMER = "customer_escalation_policy.md"
FINANCE = "financial_approval_policy.md"

CASES: list[dict[str, Any]] = [
    case("direct_01", "direct", "What is the standard shipment delivery SLA?", [evidence(SLA, "standard delivery SLA is 24 hours")], [["24 hours"], ["dispatch confirmation"]], queries("What is the standard shipment delivery SLA?", "standard shipment SLA duration", "24 hour delivery service level", "when standard SLA clock starts")),
    case("direct_02", "direct", "What is the priority customer delivery SLA?", [evidence(SLA, "priority customer delivery SLA is 18 hours")], [["18 hours"], ["dispatch confirmation"]], queries("What is the priority customer delivery SLA?", "priority shipment SLA duration", "18 hour priority service level", "priority SLA dispatch confirmation")),
    case("direct_03", "direct", "When is a shipment considered at risk?", [evidence(SLA, "predicted arrival is more than 2 hours later")], [["more than 2 hours", "exceeds 2 hours"], ["planned arrival"]], queries("When is a shipment considered at risk?", "at risk delay classification threshold", "predicted arrival versus planned arrival", "two hour shipment risk rule")),
    case("direct_04", "direct", "How quickly must a dispatcher contact the carrier for an SLA-critical exception, and when must the carrier respond?", [evidence(EXCEPTION, "contact the carrier within 15 minutes")], [["15 minutes"], ["30 minutes"], ["revised arrival estimate"], ["corrective action"]], queries("How quickly must a dispatcher contact the carrier for an SLA-critical exception, and when must the carrier respond?", "SLA critical carrier contact deadline", "carrier revised arrival estimate response time", "carrier corrective action within 30 minutes")),
    case("direct_05", "direct", "Which checks must a dispatcher complete before dispatch?", [evidence(DISPATCH, "confirm the delivery address")], [["delivery address"], ["shipment priority"], ["assigned carrier"], ["carrier has accepted the route"]], queries("Which checks must a dispatcher complete before dispatch?", "pre-dispatch checks address priority carrier", "carrier route acceptance before dispatch", "customs dangerous goods documents check")),
    case("direct_06", "direct", "How often must a priority customer receive updates during an active SLA-critical incident?", [evidence(CUSTOMER, "receive an update at least every hour")], [["every hour", "hourly"], ["priority"]], queries("How often must a priority customer receive updates during an active SLA-critical incident?", "priority customer incident communication frequency", "hourly customer updates SLA critical", "material change immediate customer communication")),
    case("direct_07", "direct", "Who approves an unplanned operational cost of EUR 6,000?", [evidence(FINANCE, "above EUR 5,000")], [["European Finance Director"], ["EUR 5,000", "5,000"]], queries("Who approves an unplanned operational cost of EUR 6,000?", "EUR 6000 unplanned cost approver", "cost above EUR 5000 approval", "European Finance Director approval threshold")),
    case("direct_08", "direct", "When is a root-cause review required and by when?", [evidence(EXCEPTION, "SLA breach of more than 4 hours")], [["more than 4 hours"], ["2 business days"]], queries("When is a root-cause review required and by when?", "root cause review SLA breach threshold", "four hour breach investigation", "root cause review within two business days")),

    case("paraphrased_01", "paraphrased", "At what event does the delivery clock begin running?", [evidence(DISPATCH, "timestamp starts the delivery SLA clock")], [["dispatch confirmation"], ["left the origin facility"]], queries("At what event does the delivery clock begin running?", "event starting delivery SLA clock", "dispatch confirmation timestamp SLA", "shipment leaves origin facility confirmation")),
    case("paraphrased_02", "paraphrased", "How long can live tracking be silent before dispatch investigates?", [evidence(DISPATCH, "tracking data has not been received for 60 minutes")], [["60 minutes"], ["investigate"]], queries("How long can live tracking be silent before dispatch investigates?", "missing tracking data investigation threshold", "no tracking update for one hour", "dispatcher monitoring silence")),
    case("paraphrased_03", "paraphrased", "May one large commitment be divided to stay below an approval limit?", [evidence(FINANCE, "Splitting one financial commitment")], [["prohibited", "may not", "must not"]], queries("May one large commitment be divided to stay below an approval limit?", "split financial commitment approval threshold", "avoid approval limit rule", "financial commitment anti splitting policy")),
    case("paraphrased_04", "paraphrased", "How should the first customer message describe a cause that has not been verified?", [evidence(CUSTOMER, "Unverified causes must be described as under investigation")], [["under investigation"]], queries("How should the first customer message describe a cause that has not been verified?", "unverified incident cause customer wording", "first customer update unknown cause", "describe cause under investigation")),
    case("paraphrased_05", "paraphrased", "What evidence allows a delivery exception to be closed?", [evidence(EXCEPTION, "proof of delivery is recorded")], [["proof of delivery"], ["formally cancelled", "formal cancellation"]], queries("What evidence allows a delivery exception to be closed?", "conditions to close delivery exception", "proof of delivery incident closure", "formally cancelled shipment exception")),
    case("paraphrased_06", "paraphrased", "When does a normal customer's delay require the customer owner to be alerted?", [evidence(CUSTOMER, "standard customer shipment")], [["Customer Success Manager"], ["exceed 2 hours", "exceeds 2 hours"]], queries("When does a normal customer's delay require the customer owner to be alerted?", "standard customer CSM notification threshold", "standard shipment SLA breach exceeds two hours", "customer owner alert for normal shipment")),
    case("paraphrased_07", "paraphrased", "How large a predicted-arrival change triggers a transport record refresh?", [evidence(SLA, "changes by 30 minutes or more")], [["30 minutes"], ["update"]], queries("How large a predicted-arrival change triggers a transport record refresh?", "predicted arrival record update threshold", "thirty minute ETA change", "transport management delay record refresh")),

    case("cross_01", "cross_document", "A standard shipment is predicted 5 hours late. What escalation and transport-system records are required?", [evidence(SLA, "standard shipment must be escalated"), evidence(SLA, "dispatcher must record the delay reason")], [["Regional Operations Manager"], ["delay reason"], ["corrective action"], ["next review time"]], queries("A standard shipment is predicted 5 hours late. What escalation and transport-system records are required?", "standard shipment five hour delay escalation", "Regional Operations Manager four hour threshold", "transport system delay records corrective action owner review")),
    case("multi_01_financial_threshold_regression", "cross_document", "A priority customer shipment is expected to arrive 5 hours late and carries a EUR 1,500 contractual penalty. What actions are required?", [evidence(SLA, "priority customer shipment must be escalated"), evidence(CUSTOMER, "notify the Customer Success Manager within 30 minutes"), evidence(FINANCE, "penalties above EUR 1,000")], [["Regional Operations Manager"], ["Customer Success Manager"], ["30 minutes"], ["before the SLA expires"], ["EUR 1,000", "1,000"]], queries("A priority customer shipment is expected to arrive 5 hours late and carries a EUR 1,500 contractual penalty. What actions are required?", "priority shipment five hour delay escalation threshold", "priority incident Customer Success Manager notification deadline", "EUR 1500 contractual penalty approval threshold approver")),
    case("cross_03", "cross_document", "A priority shipment has a vehicle breakdown, is predicted 3 hours late, and will miss its SLA. Who owns the escalated incident, what carrier response is required, and how often must it be reviewed?", [evidence(SLA, "priority customer shipment must be escalated"), evidence(EXCEPTION, "carrier must provide a revised arrival estimate"), evidence(EXCEPTION, "priority customer shipments must be reviewed")], [["Regional Operations Manager"], ["30 minutes"], ["revised arrival estimate"], ["corrective action"], ["every 30 minutes"]], queries("A priority shipment has a vehicle breakdown, is predicted 3 hours late, and will miss its SLA. Who owns the escalated incident, what carrier response is required, and how often must it be reviewed?", "priority shipment three hour delay escalation incident owner", "vehicle breakdown carrier revised estimate corrective action deadline", "SLA critical priority operational review frequency")),
    case("cross_04", "cross_document", "An expedited transport option costs EUR 6,000 and Customer Success wants to offer compensation. What approvals and communication restrictions apply?", [evidence(FINANCE, "above EUR 5,000"), evidence(FINANCE, "may not make a binding offer"), evidence(CUSTOMER, "must not be promised compensation")], [["European Finance Director"], ["binding offer"], ["approval"]], queries("An expedited transport option costs EUR 6,000 and Customer Success wants to offer compensation. What approvals and communication restrictions apply?", "EUR 6000 expedited transport approval", "Customer Success compensation binding offer restriction", "customer compensation promise before financial approval")),
    case("cross_05", "cross_document", "An at-risk shipment remains unresolved at shift change and later becomes SLA-critical. What handover and customer-notification records are required?", [evidence(DISPATCH, "included in the written shift handover"), evidence(CUSTOMER, "notifications must be recorded")], [["written shift handover"], ["customer relationship management system", "CRM"], ["timestamp"], ["next update"]], queries("An at-risk shipment remains unresolved at shift change and later becomes SLA-critical. What handover and customer-notification records are required?", "unresolved at risk shipment shift handover fields", "customer notification CRM record fields", "handover owner next review and promised next update")),
    case("cross_06", "cross_document", "For an SLA-critical priority shipment, how often are operational reviews and customer updates required?", [evidence(EXCEPTION, "priority customer shipments must be reviewed at least every 30 minutes"), evidence(CUSTOMER, "receive an update at least every hour")], [["30 minutes"], ["every hour", "hourly"]], queries("For an SLA-critical priority shipment, how often are operational reviews and customer updates required?", "priority shipment operational review every 30 minutes", "priority customer update every hour", "difference operational review and external update frequency")),
    case("cross_07", "cross_document", "A breach lasts 6 hours and creates a potential EUR 1,500 penalty. What review and approval are required?", [evidence(EXCEPTION, "root-cause review within 2 business days"), evidence(FINANCE, "penalties above EUR 1,000")], [["root-cause review"], ["2 business days"], ["Regional Operations Manager"]], queries("A breach lasts 6 hours and creates a potential EUR 1,500 penalty. What review and approval are required?", "six hour SLA breach root cause review", "root cause within two business days", "EUR 1500 potential penalty approval Regional Operations Manager")),

    case("ambiguous_01", "ambiguous", "What happens once the delay passes the normal escalation line?", [evidence(SLA, "standard shipment must be escalated")], [["Regional Operations Manager"], ["4 hours"]], queries("What happens once the delay passes the normal escalation line?", "standard shipment escalation threshold", "normal shipment predicted delay over four hours", "who receives standard shipment escalation")),
    case("ambiguous_02", "ambiguous", "Who takes over after the escalation?", [evidence(EXCEPTION, "Regional Operations Manager becomes the incident owner")], [["Regional Operations Manager"], ["incident owner"]], queries("Who takes over after the escalation?", "incident ownership after shipment escalation", "Regional Operations Manager becomes owner", "delivery exception formal transfer ownership")),
    case("ambiguous_03", "ambiguous", "What must go into the first message?", [evidence(CUSTOMER, "first customer update must state")], [["shipment status"], ["known cause"], ["revised arrival estimate"], ["corrective action"], ["next update"]], queries("What must go into the first message?", "first customer incident update required content", "shipment status cause revised arrival corrective action", "customer message next update time")),
    case("ambiguous_04", "ambiguous", "When do we check it again for a VIP load?", [evidence(EXCEPTION, "priority customer shipments must be reviewed at least every 30 minutes")], [["30 minutes"]], queries("When do we check it again for a VIP load?", "priority customer shipment operational review frequency", "SLA critical priority review interval", "thirty minute incident review")),
    case("ambiguous_05", "ambiguous", "Can urgency bypass the sign-off?", [evidence(FINANCE, "Urgency does not remove the approval requirement")], [["does not", "cannot", "no"], ["approval"]], queries("Can urgency bypass the sign-off?", "urgent expedited transport approval requirement", "urgency financial approval exception", "does urgency remove cost approval")),
    case("ambiguous_06", "ambiguous", "What if the ETA moves again?", [evidence(SLA, "predicted arrival changes by 30 minutes or more")], [["30 minutes"], ["update"]], queries("What if the ETA moves again?", "predicted arrival change record update", "ETA changes thirty minutes transport system", "delay record update rule")),
    case("ambiguous_07", "ambiguous", "Who is allowed to speak to the customer?", [evidence(CUSTOMER, "owns external communication")], [["Customer Success Manager"], ["account plan"]], queries("Who is allowed to speak to the customer?", "owner of external customer communication", "Customer Success Manager communication owner", "customer account plan named owner")),

    case("unsupported_01", "unsupported", "How many trucks does Atlas Logistics own in Germany?", [], [], queries("How many trucks does Atlas Logistics own in Germany?", "Atlas Germany fleet count", "number of owned trucks", "German vehicle inventory")),
    case("unsupported_02", "unsupported", "How many warehouse facilities does Atlas operate in Spain?", [], [], queries("How many warehouse facilities does Atlas operate in Spain?", "Atlas Spain warehouse count", "Spanish facility locations", "number of warehouses Spain")),
    case("unsupported_03", "unsupported", "What is the employee vacation allowance?", [], [], queries("What is the employee vacation allowance?", "Atlas annual leave policy", "employee holiday entitlement", "vacation days")),
    case("unsupported_04", "unsupported", "Which carrier handled the most shipments in France last quarter?", [], [], queries("Which carrier handled the most shipments in France last quarter?", "France top carrier volume", "quarterly carrier ranking", "shipment count by carrier")),
    case("unsupported_05", "unsupported", "What percentage of deliveries arrived on time last quarter?", [], [], queries("What percentage of deliveries arrived on time last quarter?", "last quarter on time delivery KPI", "historical SLA performance percentage", "delivery performance statistics")),
    case("unsupported_06", "unsupported", "What contractual penalty applies automatically to every missed SLA?", [], [], queries("What contractual penalty applies automatically to every missed SLA?", "automatic penalty amount SLA breach", "contractual penalty formula", "missed delivery compensation amount")),
    case("unsupported_07", "unsupported", "What is the maximum weight for dangerous-goods shipments?", [], [], queries("What is the maximum weight for dangerous-goods shipments?", "dangerous goods maximum shipment weight", "hazardous cargo weight limit", "dangerous goods transport restriction")),
    case("unsupported_08", "unsupported", "Which form number is used for the root-cause review?", [], [], queries("Which form number is used for the root-cause review?", "root cause review form identifier", "two business day review template", "incident investigation form number")),

    case("authorization_01", "authorization", "Under employee access, what is the standard delivery SLA?", [evidence(SLA, "standard delivery SLA is 24 hours")], [["24 hours"]], queries("Under employee access, what is the standard delivery SLA?", "standard delivery SLA employee policy", "shipment SLA duration", "dispatch confirmation 24 hours")),
    case("authorization_02", "authorization", "Under employee access, who owns external customer communication?", [evidence(CUSTOMER, "owns external communication")], [["Customer Success Manager"]], queries("Under employee access, who owns external customer communication?", "customer communication owner policy", "Customer Success Manager external communication", "account plan named owner")),
    case("authorization_03", "authorization", "Under employee access, what is the carrier response deadline?", [evidence(EXCEPTION, "within 30 minutes of contact")], [["30 minutes"]], queries("Under employee access, what is the carrier response deadline?", "carrier response deadline exception", "revised arrival estimate corrective action", "thirty minutes after contact")),
    case("authorization_04", "authorization", "Under employee access, who approves a EUR 12,000 potential penalty?", [evidence(FINANCE, "above EUR 10,000")], [["Regional Operations Manager"], ["European Finance Director"]], queries("Under employee access, who approves a EUR 12,000 potential penalty?", "EUR 12000 penalty approval", "penalty above EUR 10000 approvers", "Regional Operations Manager European Finance Director")),
    case("authorization_05", "authorization", "Under employee access, what is required at shift handover?", [evidence(DISPATCH, "included in the written shift handover")], [["shipment"], ["current status"], ["next action"], ["owner"], ["next review time"]], queries("Under employee access, what is required at shift handover?", "written shift handover required fields", "unresolved at risk shipment handover", "status action owner next review")),
    case("authorization_06", "authorization", "Under employee access, how is approved compensation recorded?", [evidence(FINANCE, "recorded in both the customer relationship management system")], [["customer relationship management system", "CRM"], ["incident record"]], queries("Under employee access, how is approved compensation recorded?", "approved compensation recording systems", "CRM and incident record", "financial approval compensation records")),

    case("conversation_01", "conversational", "And when should they be escalated?", [evidence(SLA, "priority customer shipment must be escalated")], [["2 hours"], ["Regional Operations Manager"]], queries("When should priority customer shipments be escalated?", "priority shipment escalation threshold", "predicted delay exceeds two hours", "Regional Operations Manager priority escalation"), history=[{"role": "user", "content": "What is the SLA for priority customer shipments?"}, {"role": "assistant", "content": "The priority customer delivery SLA is 18 hours from dispatch confirmation."}]),
    case("conversation_02", "conversational", "Who needs to be told, and by when?", [evidence(CUSTOMER, "notify the Customer Success Manager within 30 minutes")], [["Customer Success Manager"], ["30 minutes"], ["before the SLA expires"]], queries("Who must be notified when a priority customer shipment risks missing its SLA, and by when?", "priority incident Customer Success Manager notification", "notify within 30 minutes after risk", "notification before SLA expiry"), history=[{"role": "user", "content": "A priority customer shipment is now at risk of missing delivery."}, {"role": "assistant", "content": "The incident should be managed under priority customer rules."}]),
    case("conversation_03", "conversational", "What are the two update frequencies?", [evidence(EXCEPTION, "priority customer shipments must be reviewed at least every 30 minutes"), evidence(CUSTOMER, "receive an update at least every hour")], [["30 minutes"], ["every hour", "hourly"]], queries("What are the operational review and customer update frequencies for an SLA-critical priority shipment?", "priority operational review every 30 minutes", "priority customer communication every hour", "two update frequencies SLA critical priority"), history=[{"role": "user", "content": "Consider an SLA-critical priority customer shipment."}, {"role": "assistant", "content": "We need both operational reviews and customer communications."}]),
    case("conversation_04", "conversational", "What does she need before offering it?", [evidence(FINANCE, "may not make a binding offer without approval")], [["approval"], ["binding offer"]], queries("What approval does the Customer Success Manager need before offering compensation?", "Customer Success Manager binding compensation offer", "financial approval before customer compensation", "credit proposal approval requirement"), history=[{"role": "user", "content": "The Customer Success Manager wants to offer the customer a credit."}, {"role": "assistant", "content": "The Financial Approval Policy applies."}]),
    case("conversation_05", "conversational", "Does that end their responsibility?", [evidence(DISPATCH, "remains responsible until the incoming dispatcher acknowledges")], [["incoming dispatcher acknowledges"], ["outgoing dispatcher"]], queries("Does shift handover end the outgoing dispatcher's responsibility?", "outgoing dispatcher responsibility handover", "incoming dispatcher acknowledgement", "when handover responsibility transfers"), history=[{"role": "user", "content": "The outgoing dispatcher added the at-risk shipment to the written handover."}, {"role": "assistant", "content": "The incoming shift has not acknowledged it yet."}]),
    case("conversation_06", "conversational", "What should happen next if it changes materially?", [evidence(CUSTOMER, "must be communicated without waiting")], [["without waiting"], ["communicated"]], queries("What happens when the arrival estimate or corrective action changes materially during a customer incident?", "material arrival estimate change customer communication", "do not wait for scheduled update", "corrective action change immediate update"), history=[{"role": "user", "content": "The priority customer is receiving hourly updates."}, {"role": "assistant", "content": "A new route has materially changed the arrival estimate."}]),
    case("conversation_07", "conversational", "Who approves the higher amount too?", [evidence(FINANCE, "above EUR 10,000 also require approval")], [["European Finance Director"], ["Regional Operations Manager"]], queries("Who approves a potential contractual penalty above EUR 10,000?", "penalty above EUR 10000 additional approver", "European Finance Director penalty approval", "Regional Operations Manager high penalty"), history=[{"role": "user", "content": "The Regional Operations Manager approves penalties above EUR 1,000."}, {"role": "assistant", "content": "A larger threshold adds another approval."}]),
]


def validate_cases(cases: list[dict[str, Any]] = CASES) -> None:
    required_categories = {
        "direct", "paraphrased", "cross_document", "ambiguous",
        "unsupported", "authorization", "conversational",
    }
    if len(cases) != 50:
        raise ValueError(f"Expected exactly 50 cases, found {len(cases)}.")
    ids = [item["id"] for item in cases]
    if len(ids) != len(set(ids)):
        raise ValueError("Checkpoint 11 case IDs must be unique.")
    if {item["category"] for item in cases} != required_categories:
        raise ValueError("Checkpoint 11 category coverage is incomplete.")
    for item in cases:
        if len(item["search_queries"]) != 4:
            raise ValueError(f"{item['id']} must have four fixed search queries.")
        unsupported = item["category"] == "unsupported"
        if unsupported == bool(item["expected_evidence"]):
            raise ValueError(f"{item['id']} has inconsistent evidence labels.")
        if not unsupported and not item["expected_facts"]:
            raise ValueError(f"{item['id']} has no expected answer facts.")


validate_cases()
