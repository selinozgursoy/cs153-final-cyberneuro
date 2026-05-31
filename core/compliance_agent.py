"""
CyberNeuro — BCI Vendor Compliance Agent
Autonomously audits BCI vendor privacy policies against neurorights laws.

Laws checked:
  - Colorado HB 24-1058 (neural data as sensitive biological data)
  - California SB 1223 (CCPA amendment for neural data)
  - Minnesota HF 1 (data minimization)
  - EU AI Act Article 5 (no subliminal manipulation)
  - General neurorights standards (Neurorights Foundation 2024)

Supports: OpenRouter API (preferred) or Anthropic API directly
"""

import os
import json
from dataclasses import dataclass, field
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

COMPLIANCE_RUBRIC = [
    {"id": "CO-01", "law": "Colorado HB 24-1058",
     "requirement": "Explicit opt-in consent before collecting neural data",
     "check": "Does the policy require opt-in (not opt-out) consent before any neural data collection?",
     "severity": "CRITICAL"},
    {"id": "CO-02", "law": "Colorado HB 24-1058",
     "requirement": "Neural data classified as sensitive biological data",
     "check": "Does the policy classify neural/brain data as sensitive data with heightened protections?",
     "severity": "HIGH"},
    {"id": "CA-01", "law": "California SB 1223",
     "requirement": "Right to delete neural data",
     "check": "Does the policy grant users a clear right to delete their neural data?",
     "severity": "HIGH"},
    {"id": "CA-02", "law": "California SB 1223",
     "requirement": "No sale of neural data without explicit consent",
     "check": "Does the policy explicitly prohibit selling neural data without separate explicit consent?",
     "severity": "CRITICAL"},
    {"id": "MN-01", "law": "Minnesota HF 1",
     "requirement": "Data minimization for neural signals",
     "check": "Does the policy commit to collecting only minimum neural data necessary?",
     "severity": "MEDIUM"},
    {"id": "EU-01", "law": "EU AI Act Article 5",
     "requirement": "No subliminal manipulation via neural interface",
     "check": "Does the policy prohibit using the BCI to influence users through subliminal techniques?",
     "severity": "CRITICAL"},
    {"id": "EU-02", "law": "EU AI Act / GDPR",
     "requirement": "Encryption of neural data in transit and at rest",
     "check": "Does the policy state that neural data is encrypted both in transmission and storage?",
     "severity": "HIGH"},
    {"id": "GEN-01", "law": "Neurorights Foundation Standards",
     "requirement": "Meaningful third-party data sharing limitations",
     "check": "Does the policy meaningfully limit which third parties can access neural data?",
     "severity": "HIGH"},
    {"id": "GEN-02", "law": "Neurorights Foundation Standards",
     "requirement": "Breach notification for neural data",
     "check": "Does the policy commit to notifying users within a specific timeframe if neural data is compromised?",
     "severity": "HIGH"},
    {"id": "GEN-03", "law": "Neurorights Foundation Standards",
     "requirement": "No indefinite retention of raw neural signals",
     "check": "Does the policy specify a maximum retention period for raw neural data?",
     "severity": "MEDIUM"},
]

DEMO_POLICIES = {
    "NeuroCorp (Poor Compliance Demo)": """
    By using our EEG headset, you agree to our collection of brain activity data.
    We collect neural signals to improve our products and may share this data with
    our advertising partners and research affiliates. You may opt out of marketing
    communications at any time. Data is retained for as long as necessary to fulfill
    our business purposes. We implement reasonable security measures. Neural data may
    be transferred internationally. We reserve the right to sell anonymized neural
    datasets to third parties for research purposes.
    """,
    "NeuroSafe (Strong Compliance Demo)": """
    We collect neural data ONLY with your explicit opt-in consent, obtained separately
    before any recording begins. Neural signals are classified as sensitive biological
    data with the highest tier of protection. We never sell your neural data under any
    circumstances. All neural data is encrypted using AES-256 at rest and TLS 1.3 in
    transit. You may request complete deletion of your neural data at any time, processed
    within 30 days. We retain raw neural signals for a maximum of 90 days. Third parties
    accessing derived data must agree to equivalent protections. In the event of a breach
    affecting your neural data, we will notify you within 72 hours.
    """,
}


@dataclass
class ComplianceViolation:
    requirement_id: str
    law: str
    requirement: str
    verdict: str
    severity: str
    evidence: str
    recommendation: str


@dataclass
class ComplianceReport:
    vendor_name: str
    policy_source: str
    violations: list = field(default_factory=list)
    overall_score: float = 0.0
    risk_level: str = "UNKNOWN"

    @property
    def n_critical(self):
        return sum(1 for v in self.violations
                   if v.severity == "CRITICAL" and v.verdict != "COMPLIANT")
    @property
    def n_compliant(self):
        return sum(1 for v in self.violations if v.verdict == "COMPLIANT")


class ComplianceAgent:
    def __init__(self, api_key=None):
        self.api_key = (api_key or
                        os.environ.get("OPENROUTER_API_KEY") or
                        os.environ.get("ANTHROPIC_API_KEY"))
        if not self.api_key:
            raise ValueError(
                "No API key found. Set OPENROUTER_API_KEY or ANTHROPIC_API_KEY."
            )

    def _call_api(self, prompt):
        import urllib.request
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }
        body = json.dumps({
            "model": "anthropic/claude-sonnet-4-5",
            "max_tokens": 300,
            "messages": [{"role": "user", "content": prompt}],
        }).encode()

        # Try OpenRouter first
        for url in ["https://openrouter.ai/api/v1/chat/completions",
                    "https://api.anthropic.com/v1/messages"]:
            try:
                req = urllib.request.Request(url, data=body, headers=headers, method="POST")
                if "anthropic.com" in url:
                    headers["x-api-key"] = self.api_key
                    headers["anthropic-version"] = "2023-06-01"
                    body_dict = {"model": "claude-sonnet-4-20250514", "max_tokens": 300,
                                 "messages": [{"role": "user", "content": prompt}]}
                    body = json.dumps(body_dict).encode()
                    req = urllib.request.Request(url, data=body, headers=headers, method="POST")

                with urllib.request.urlopen(req, timeout=15) as resp:
                    data = json.loads(resp.read())
                    if "choices" in data:
                        return data["choices"][0]["message"]["content"].strip()
                    elif "content" in data:
                        return data["content"][0]["text"].strip()
            except Exception:
                continue
        raise RuntimeError("Both OpenRouter and Anthropic API failed")

    def _check_requirement(self, policy_text, requirement):
        prompt = f"""You are a legal compliance analyst for neurotechnology privacy law.

Analyze this BCI vendor privacy policy against one requirement:

REQUIREMENT ID: {requirement['id']}
LAW: {requirement['law']}
REQUIREMENT: {requirement['requirement']}
CHECK: {requirement['check']}

PRIVACY POLICY:
\"\"\"{policy_text[:6000]}\"\"\"

Respond ONLY with JSON (no markdown):
{{"verdict": "COMPLIANT|VIOLATION|UNCLEAR|NOT_MENTIONED", "evidence": "quote or paraphrase from policy", "recommendation": "what vendor should do to fix this"}}"""

        try:
            raw = self._call_api(prompt)
            if "```" in raw:
                raw = raw.split("```")[1].lstrip("json").strip()
            parsed = json.loads(raw)
        except Exception:
            parsed = {"verdict": "UNCLEAR", "evidence": "Analysis failed",
                      "recommendation": "Manual review required"}

        return ComplianceViolation(
            requirement_id=requirement["id"],
            law=requirement["law"],
            requirement=requirement["requirement"],
            verdict=parsed.get("verdict", "UNCLEAR"),
            severity=requirement["severity"],
            evidence=parsed.get("evidence", ""),
            recommendation=parsed.get("recommendation", ""),
        )

    def audit(self, policy_text, vendor_name="Unknown", policy_source="provided"):
        console.print(Panel.fit(
            f"[bold purple]CyberNeuro[/] Compliance Agent\n"
            f"Auditing: [bold]{vendor_name}[/]\n"
            f"Checking {len(COMPLIANCE_RUBRIC)} neurorights requirements...",
            border_style="purple",
        ))

        violations = []
        for i, req in enumerate(COMPLIANCE_RUBRIC, 1):
            console.print(f"  [{i:2d}/{len(COMPLIANCE_RUBRIC)}] {req['id']}: "
                          f"{req['requirement'][:55]}...")
            v = self._check_requirement(policy_text, req)
            violations.append(v)
            color = {"COMPLIANT": "green", "VIOLATION": "red",
                     "UNCLEAR": "yellow", "NOT_MENTIONED": "orange1"}.get(v.verdict, "white")
            console.print(f"         → [{color}]{v.verdict}[/]")

        # Score calculation
        weights = {"CRITICAL": 20, "HIGH": 10, "MEDIUM": 5}
        penalty = sum(
            weights.get(v.severity, 3) * (1.0 if v.verdict == "VIOLATION" else 0.5)
            for v in violations if v.verdict != "COMPLIANT"
        )
        score = max(0.0, 100.0 - penalty)
        risk  = ("CRITICAL" if score < 40 else "HIGH" if score < 60 else
                 "MEDIUM" if score < 80 else "LOW")

        report = ComplianceReport(
            vendor_name=vendor_name, policy_source=policy_source,
            violations=violations, overall_score=round(score, 1), risk_level=risk,
        )
        self._print_report(report)
        return report

    def _print_report(self, report):
        colors = {"CRITICAL": "bold red", "HIGH": "red", "MEDIUM": "yellow", "LOW": "green"}
        console.print(Panel(
            f"[bold]Vendor:[/] {report.vendor_name}\n"
            f"[bold]Score:[/]  {report.overall_score}/100\n"
            f"[bold]Risk:[/]   [{colors.get(report.risk_level,'white')}]{report.risk_level}[/]\n"
            f"[bold]Critical violations:[/] [red]{report.n_critical}[/] | "
            f"[bold]Compliant:[/] [green]{report.n_compliant}/{len(COMPLIANCE_RUBRIC)}[/]",
            title="[bold purple]CyberNeuro Compliance Report[/]",
            border_style="purple",
        ))

        t = Table(style="purple", show_lines=True)
        t.add_column("ID", width=7); t.add_column("Severity", width=9)
        t.add_column("Requirement", width=30); t.add_column("Verdict", width=14)
        t.add_column("Evidence", width=40)

        order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2}
        for v in sorted(report.violations, key=lambda x: (order.get(x.severity,9), x.verdict=="COMPLIANT")):
            vc = {"COMPLIANT":"green","VIOLATION":"red","UNCLEAR":"yellow","NOT_MENTIONED":"orange1"}.get(v.verdict,"white")
            sc = {"CRITICAL":"bold red","HIGH":"red","MEDIUM":"yellow"}.get(v.severity,"white")
            t.add_row(v.requirement_id, f"[{sc}]{v.severity}[/]",
                      v.requirement[:28], f"[{vc}]{v.verdict}[/]",
                      v.evidence[:60] if v.verdict != "COMPLIANT" else f"[dim]{v.evidence[:60]}[/]")
        console.print(t)

    def export_json(self, report, path):
        data = {
            "vendor": report.vendor_name,
            "score": report.overall_score,
            "risk_level": report.risk_level,
            "n_critical": report.n_critical,
            "n_compliant": report.n_compliant,
            "requirements": [
                {"id": v.requirement_id, "law": v.law, "requirement": v.requirement,
                 "severity": v.severity, "verdict": v.verdict,
                 "evidence": v.evidence, "recommendation": v.recommendation}
                for v in report.violations
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        console.print(f"[green]✓[/] Compliance report saved to {path}")


def _demo_structure_only():
    """Show compliance rubric without making API calls."""
    console.print(Panel.fit(
        "[bold]CyberNeuro Compliance Agent[/]\n\n"
        "Checks 10 neurorights requirements:\n" +
        "\n".join(f"  [{r['severity']:8s}] {r['id']}: {r['requirement']}"
                  for r in COMPLIANCE_RUBRIC),
        title="Neurorights Compliance Rubric",
        border_style="purple",
    ))
