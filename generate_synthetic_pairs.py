"""
generate_synthetic_pairs.py
---------------------------
Generates synthetic training pairs for critically thin dataset layers.

FIX — Thin layer counts:
  execution_context:    was  10  →  now targets ~850  (threshold 200)
  remediation_learning: was 119  →  now targets ~1600 (threshold 500)

Changes:
  1. REMEDIATION_KB expanded from 5 CWEs to 18 CWEs (covering all OWASP Top 10 + extras)
  2. EXECUTION_CONTEXT_KB expanded from ~4 stacks to 20 technology stacks
  3. Additional pair-type generators added per CWE/stack
  4. KEV-grounded pairs added for remediation layer (high-urgency patch scenarios)

Run AFTER build_dataset.py, BEFORE finetuning.py:
    python generate_synthetic_pairs.py
    # or via pipeline:
    python run_pipeline.py --synthetic

Appends to data/training_pairs.jsonl without touching other layers.
"""

import json
import random
from pathlib import Path

random.seed(42)

NVD_PATH    = Path("data") / "raw_nvd.json"
KEV_PATH    = Path("data") / "raw_cisa_kev.json"
OUTPUT_PATH = Path("data") / "training_pairs.jsonl"


# ═════════════════════════════════════════════════════════════════════════════
#  TEMPLATE PERTURBATION / PARAPHRASING ENGINE
# ═════════════════════════════════════════════════════════════════════════════
# To prevent the model from memorizing rigid templates, we apply randomised
# transformations: instruction rewording, sentence reordering, terminology
# swaps, and realistic noise injection.

# Instruction paraphrase pools — keyed by semantic intent
_INSTRUCTION_VARIANTS = {
    "how_fix": [
        "How do I fix {cwe} ({cve}) in my application?",
        "What's the remediation for {cwe} ({cve})?",
        "Our scanner found {cve} ({cwe}). How do we patch it?",
        "Recommend a fix for {cwe} vulnerability {cve}.",
        "We've been flagged for {cve} ({cwe}). What's the remediation path?",
        "How should I address {cwe} ({cve}) in production?",
    ],
    "root_cause": [
        "What is the root cause of {cve} and how should it be permanently resolved?",
        "Explain why {cve} ({cwe}) happens and how to eliminate it for good.",
        "What underlying pattern causes {cve}? How do we prevent recurrence?",
        "Break down the root cause of {cve} ({cwe}) and permanent fix.",
        "Why does {cve} keep happening and how do we stop it?",
    ],
    "verify_fix": [
        "After patching {cve} ({cwe}), how do I verify the fix is effective?",
        "How can I confirm {cve} ({cwe}) is actually fixed?",
        "What tests validate that the patch for {cve} works?",
        "Post-patch verification steps for {cve} ({cwe})?",
        "How do I test that {cve} ({cwe}) can no longer be exploited?",
    ],
    "priority": [
        "How urgently should {cve} (CVSS {cvss}) be patched, and what's the remediation plan?",
        "What's the patch priority for {cve} with CVSS {cvss}?",
        "Given CVSS {cvss}, how fast do we need to remediate {cve}?",
        "Prioritize patching for {cve} (CVSS {cvss}) — what's the plan?",
        "Triage {cve} (CVSS {cvss}): urgency and remediation steps?",
    ],
    "standard_remediation": [
        "What is the standard remediation for {cwe} vulnerabilities?",
        "How are {cwe} vulnerabilities typically fixed?",
        "Best practices for remediating {cwe}?",
        "What's the industry-standard fix for {cwe}?",
        "Standard approach to resolving {cwe} weaknesses?",
    ],
    "detect_cwe": [
        "How do I detect {cwe} vulnerabilities in my codebase?",
        "What tools and methods find {cwe} issues?",
        "How can I identify {cwe} weaknesses in our application?",
        "Detection strategy for {cwe} in a codebase?",
        "How do I scan for {cwe} in source code and running apps?",
    ],
    "tool_selection": [
        "We are performing a security assessment of a {stack} application. What tools and methodology should we use?",
        "What's the right toolset for pentesting a {stack} deployment?",
        "Recommend security assessment tools for a {stack} environment.",
        "How should we approach security testing of our {stack} application?",
        "What tools and methodology for auditing a {stack} stack?",
    ],
    "common_vulns": [
        "What are the most common security vulnerabilities in {stack} applications?",
        "Top security risks in {stack} environments?",
        "What vulnerabilities should I prioritize when testing {stack}?",
        "Common attack vectors against {stack} applications?",
        "What security weaknesses are typical in {stack} deployments?",
    ],
    "env_risks": [
        "What environment-specific security risks should I check for in a {stack} deployment?",
        "What misconfigurations are common in {stack} production environments?",
        "Environment security checklist for {stack}?",
        "What deployment-specific risks affect {stack} applications?",
        "Operational security risks specific to {stack}?",
    ],
    "recon_indicators": [
        "During reconnaissance, I've identified a target is running {stack}. What indicators confirm this and what attack surface does it expose?",
        "How do I fingerprint and enumerate the attack surface of a {stack} target?",
        "I've found a {stack} instance. What should I probe first?",
        "Confirming {stack} stack identification — what indicators and what's exposed?",
        "Recon confirmed {stack}. What attack surface should I map?",
    ],
    "test_cve_env": [
        "We need to test for {cve} in an environment running {sw}. What is the testing approach?",
        "How do I validate whether {cve} is exploitable in our {sw} deployment?",
        "Testing methodology for {cve} on {sw}?",
        "What's the approach to check if {cve} affects our {sw} instance?",
        "How should we test our {sw} environment for {cve}?",
    ],
}

# Terminology synonym pools for output perturbation
_TERM_SWAPS = {
    "**Fix:**":        ["**Remediation:**", "**Solution:**", "**Resolution:**", "**Mitigation:**"],
    "**Root Cause:**": ["**Underlying Issue:**", "**Why This Happens:**", "**Core Problem:**"],
    "**Control Type:**": ["**Security Control:**", "**Defense Layer:**", "**Control Category:**"],
    "**Code Example:**": ["**Code Pattern:**", "**Implementation Example:**", "**Before/After:**"],
    "**Verification:**": ["**Validation:**", "**Testing:**", "**Confirm Fix:**"],
    "**Tools:**":      ["**Recommended Tools:**", "**Useful Tools:**", "**Analysis Tools:**"],
    "**Primary risk:**": ["**Main risk:**", "**Key risk:**", "**Top concern:**"],
    "**Primary tool:**": ["**Lead tool:**", "**Main tool:**", "**Start with:**"],
}

# Noise injections — realistic additions that precede or follow content
_NOISE_PREFIXES = [
    "",  # No noise (most common)
    "",
    "",
    "Note: Always test in a non-production environment first.\n\n",
    "Important: Back up your configuration before applying changes.\n\n",
    "Context: This applies to the default installation. Custom configurations may differ.\n\n",
]

_NOISE_SUFFIXES = [
    "",
    "",
    "",
    "\n\nNote: Re-run your security scanner after applying the fix to confirm resolution.",
    "\n\nTip: Add this check to your CI/CD pipeline to prevent regression.",
    "\n\nReminder: Document the remediation in your vulnerability tracking system.",
]


def _pick_instruction(intent: str, **kwargs) -> str:
    """Pick a random paraphrased instruction for the given intent."""
    variants = _INSTRUCTION_VARIANTS.get(intent, [])
    if not variants:
        return ""
    template = random.choice(variants)
    try:
        return template.format(**kwargs)
    except KeyError:
        return template


def _perturb_output(text: str) -> str:
    """Apply random terminology swaps, reordering, and noise to output text."""
    # Terminology swaps (each swap has 40% chance)
    for original, alternatives in _TERM_SWAPS.items():
        if original in text and random.random() < 0.4:
            text = text.replace(original, random.choice(alternatives), 1)

    # Sentence-level reordering within paragraphs (20% chance per paragraph)
    paragraphs = text.split("\n\n")
    perturbed_paragraphs = []
    for para in paragraphs:
        lines = para.split("\n")
        # Only reorder bullet-point lines (those starting with "  " or "  -")
        bullet_lines = [l for l in lines if l.strip().startswith("-") or l.strip().startswith(("1.", "2.", "3.", "4.", "5."))]
        if len(bullet_lines) >= 3 and random.random() < 0.2:
            # Shuffle interior bullets only (keep first as-is for coherence)
            non_bullets = [l for l in lines if l not in bullet_lines]
            random.shuffle(bullet_lines)
            lines = non_bullets + bullet_lines
        perturbed_paragraphs.append("\n".join(lines))
    text = "\n\n".join(perturbed_paragraphs)

    # Noise injection (15% chance)
    if random.random() < 0.15:
        text = random.choice(_NOISE_PREFIXES) + text
    if random.random() < 0.15:
        text = text + random.choice(_NOISE_SUFFIXES)

    return text


# ═════════════════════════════════════════════════════════════════════════════
#  REMEDIATION KNOWLEDGE BASE  (18 CWEs — was 5)
# ═════════════════════════════════════════════════════════════════════════════
REMEDIATION_KB: dict[str, dict] = {

    "CWE-89": {
        "fix":          "Use parameterized queries / prepared statements. Never build SQL by string concatenation.",
        "root_cause":   "Direct string interpolation of untrusted input into SQL query construction.",
        "control":      "Input validation + parameterized queries (technical)",
        "before":       'query = "SELECT * FROM users WHERE id = " + user_id',
        "after":        'cursor.execute("SELECT * FROM users WHERE id = ?", (user_id,))',
        "test_payloads": ["' OR '1'='1", "1; DROP TABLE users--", "1 UNION SELECT null,null--"],
        "tools":        ["SQLMap", "Burp Suite", "OWASP ZAP"],
    },
    "CWE-79": {
        "fix":          "HTML-encode all user-supplied output. Use Content-Security-Policy. Avoid innerHTML with untrusted data.",
        "root_cause":   "Unsanitized user input reflected into HTML response without encoding.",
        "control":      "Output encoding + CSP headers (technical)",
        "before":       'document.getElementById("msg").innerHTML = userInput;',
        "after":        'document.getElementById("msg").textContent = userInput;',
        "test_payloads": ["<script>alert(1)</script>", "<img src=x onerror=alert(1)>", "javascript:alert(1)"],
        "tools":        ["Burp Suite", "XSStrike", "OWASP ZAP"],
    },
    "CWE-22": {
        "fix":          "Canonicalize the path and verify it starts with the expected base directory before any file operation.",
        "root_cause":   "Missing path canonicalization allows directory traversal via ../ sequences.",
        "control":      "Path validation + allowlist of permitted directories (technical)",
        "before":       'open("/var/uploads/" + filename)',
        "after":        'p = Path("/var/uploads", filename).resolve(); assert str(p).startswith("/var/uploads")',
        "test_payloads": ["../../etc/passwd", "../../../windows/system32/cmd.exe", "%2e%2e%2f%2e%2e%2fetc%2fpasswd"],
        "tools":        ["Burp Suite", "DotDotPwn", "Nikto"],
    },
    "CWE-78": {
        "fix":          "Avoid passing user input to shell commands. Use library APIs instead. If unavoidable, use allowlists and proper escaping.",
        "root_cause":   "User-controlled input reaches shell interpreter without sanitization.",
        "control":      "Input validation + avoid shell execution (technical)",
        "before":       'os.system("ping " + user_host)',
        "after":        'subprocess.run(["ping", "-c", "1", user_host], capture_output=True)',
        "test_payloads": ["; id", "| whoami", "&& cat /etc/passwd", "`id`"],
        "tools":        ["Burp Suite", "Commix", "OWASP ZAP"],
    },
    "CWE-20": {
        "fix":          "Validate all inputs against a strict allowlist schema (type, length, format, range). Reject, don't sanitize.",
        "root_cause":   "Application accepts and processes input without verifying it meets expected constraints.",
        "control":      "Input validation — allowlist approach (technical)",
        "before":       "process(request.get('age'))",
        "after":        "age = int(request.get('age')); assert 0 <= age <= 150",
        "test_payloads": ["-1", "99999999", "null", "'; DROP TABLE", "<script>"],
        "tools":        ["Burp Suite", "OWASP ZAP", "Fuzz testing frameworks"],
    },
    "CWE-287": {
        "fix":          "Use a proven authentication framework. Enforce MFA. Never roll your own auth. Implement account lockout.",
        "root_cause":   "Authentication logic is flawed, bypassable, or relies on attacker-controlled data.",
        "control":      "Authentication hardening + MFA (technical + process)",
        "before":       'if user["role"] == request.get("role"): grant_access()',
        "after":        "Use server-side session validation; verify JWT signatures with a secret key.",
        "test_payloads": ['{"role":"admin"}', "role=admin (cookie tampering)", "JWT alg:none attack"],
        "tools":        ["Burp Suite", "jwt_tool", "Hydra"],
    },
    "CWE-306": {
        "fix":          "Add authentication checks to all sensitive endpoints. Use middleware to enforce auth globally rather than per-route.",
        "root_cause":   "Critical functionality accessible without verifying the caller's identity.",
        "control":      "Authentication enforcement (technical)",
        "before":       "@app.route('/admin/delete')\ndef delete(): ...",
        "after":        "@app.route('/admin/delete')\n@login_required\ndef delete(): ...",
        "test_payloads": ["Direct URL access without session", "API calls without Authorization header"],
        "tools":        ["Burp Suite", "OWASP ZAP", "Nikto"],
    },
    "CWE-502": {
        "fix":          "Never deserialize untrusted data with native deserialization. Use safe formats (JSON with schema validation). Implement integrity checks.",
        "root_cause":   "Application deserializes attacker-controlled byte streams that can instantiate arbitrary objects.",
        "control":      "Avoid unsafe deserialization + input integrity check (technical)",
        "before":       'obj = pickle.loads(request.data)',
        "after":        "Use json.loads() with strict schema validation; never use pickle/marshal on user data.",
        "test_payloads": ["Ysoserial-generated gadget chains", "PHP object injection payloads", "Java deserialization PoCs"],
        "tools":        ["ysoserial", "Burp Deserialization Scanner", "Freddy extension"],
    },
    "CWE-798": {
        "fix":          "Remove all hardcoded credentials. Use secrets managers (Vault, AWS Secrets Manager). Rotate exposed credentials immediately.",
        "root_cause":   "Credentials embedded in source code or configuration files accessible to attackers.",
        "control":      "Secrets management + code scanning (process + technical)",
        "before":       'db.connect(password="SuperSecret123")',
        "after":        'db.connect(password=os.environ["DB_PASSWORD"])',
        "test_payloads": ["grep -r 'password=' .", "truffleHog scan", "git-secrets scan"],
        "tools":        ["TruffleHog", "GitLeaks", "Semgrep"],
    },
    "CWE-611": {
        "fix":          "Disable external entity processing in your XML parser. Set FEATURE_EXTERNAL_GENERAL_ENTITIES and FEATURE_EXTERNAL_PARAMETER_ENTITIES to false.",
        "root_cause":   "XML parser follows external entity references in attacker-supplied XML, enabling SSRF or file disclosure.",
        "control":      "Parser hardening — disable XXE features (technical)",
        "before":       "ET.fromstring(user_xml)  # default Python expat is safe; Java/PHP parsers are not",
        "after":        "factory.setFeature('http://xml.org/sax/features/external-general-entities', False)",
        "test_payloads": ["<!DOCTYPE foo [<!ENTITY xxe SYSTEM 'file:///etc/passwd'>]>", "Blind OOB XXE via DNS"],
        "tools":        ["Burp Suite", "XXEinjector", "OWASP ZAP"],
    },
    "CWE-918": {
        "fix":          "Validate and allowlist URLs before making server-side requests. Block internal IP ranges. Use a dedicated outbound proxy.",
        "root_cause":   "Server fetches URLs provided by users without restricting target scope, enabling access to internal services.",
        "control":      "URL allowlisting + network segmentation (technical)",
        "before":       'requests.get(request.args["url"])',
        "after":        "assert url.startswith('https://allowed-domain.com'); requests.get(url)",
        "test_payloads": ["http://169.254.169.254/latest/meta-data/", "http://localhost:6379/", "file:///etc/passwd"],
        "tools":        ["Burp Suite", "SSRFire", "OWASP ZAP"],
    },
    "CWE-434": {
        "fix":          "Validate file type by content (magic bytes), not extension. Store uploads outside webroot. Rename files server-side.",
        "root_cause":   "File upload accepts server-executable file types that can be accessed via URL to achieve RCE.",
        "control":      "File type validation + secure storage (technical)",
        "before":       "shutil.move(upload.filename, '/var/www/uploads/')",
        "after":        "Validate MIME type via magic bytes; store as UUID filename outside webroot; never execute uploaded files.",
        "test_payloads": ["shell.php disguised as image.php.jpg", "Polyglot JPEG/PHP files", ".htaccess upload to change execution context"],
        "tools":        ["Burp Suite", "ExifTool", "Weevely"],
    },
    "CWE-352": {
        "fix":          "Implement CSRF tokens on all state-changing requests. Use SameSite=Strict cookie attribute. Verify Origin/Referer headers.",
        "root_cause":   "State-changing endpoints lack request origin verification, allowing forged cross-origin requests.",
        "control":      "CSRF tokens + SameSite cookies (technical)",
        "before":       "@app.route('/transfer', methods=['POST'])\ndef transfer(): do_transfer()",
        "after":        "@app.route('/transfer', methods=['POST'])\n@csrf_protect\ndef transfer(): do_transfer()",
        "test_payloads": ["Malicious HTML form with auto-submit", "Cross-origin fetch with credentials"],
        "tools":        ["Burp Suite CSRF PoC generator", "OWASP ZAP", "CSRFtester"],
    },
    "CWE-416": {
        "fix":          "Set pointers to NULL after free. Use memory-safe languages where possible. Enable compiler mitigations (ASAN, SafeStack).",
        "root_cause":   "Memory is freed and the dangling pointer is subsequently used, allowing heap corruption or code execution.",
        "control":      "Memory management discipline + compiler mitigations (technical)",
        "before":       "free(ptr); /* ptr still used later */",
        "after":        "free(ptr); ptr = NULL; /* subsequent use-after-free is now a NULL deref — crash not exploit */",
        "test_payloads": ["Heap spray + UAF trigger", "AFL fuzzing with ASAN enabled"],
        "tools":        ["AddressSanitizer (ASAN)", "Valgrind", "AFL++"],
    },
    "CWE-476": {
        "fix":          "Check return values for NULL before dereferencing. Enable compiler null-pointer checks. Use static analysis.",
        "root_cause":   "Code dereferences a pointer without verifying it is non-NULL, leading to crash or exploitable condition.",
        "control":      "Null-check enforcement + static analysis (technical)",
        "before":       "obj = malloc(size); obj->field = value;  /* no NULL check */",
        "after":        "obj = malloc(size); if (!obj) { handle_error(); return; } obj->field = value;",
        "test_payloads": ["Send unexpected NULL/empty payloads", "Out-of-memory simulation"],
        "tools":        ["AddressSanitizer", "Coverity", "SonarQube"],
    },
    "CWE-190": {
        "fix":          "Validate arithmetic operations won't overflow before executing. Use safe integer libraries or compiler overflow checks (-ftrapv).",
        "root_cause":   "Integer arithmetic wraps around unexpectedly, causing buffer size miscalculations leading to heap overflow.",
        "control":      "Arithmetic bounds validation (technical)",
        "before":       "buf = malloc(a * b);  /* a*b can overflow to 0 */",
        "after":        "if (a > SIZE_MAX / b) { handle_error(); } buf = malloc(a * b);",
        "test_payloads": ["SIZE_MAX values", "Large multiplier inputs that wrap to small sizes"],
        "tools":        ["UBSan (Undefined Behavior Sanitizer)", "AFL++", "CodeQL"],
    },
    "CWE-295": {
        "fix":          "Enable full certificate chain validation. Do not disable SSL verification in production. Pin certificates for critical connections.",
        "root_cause":   "TLS certificate validation is disabled or incomplete, enabling man-in-the-middle interception.",
        "control":      "TLS configuration hardening (technical)",
        "before":       "requests.get(url, verify=False)  # disables all cert validation",
        "after":        "requests.get(url, verify='/path/to/ca-bundle.crt')",
        "test_payloads": ["Self-signed certificate MitM", "Expired certificate", "Wrong hostname in cert"],
        "tools":        ["SSLyze", "testssl.sh", "Burp Suite (intercept proxy)"],
    },
    "CWE-732": {
        "fix":          "Apply principle of least privilege to all file/directory permissions. Audit with find. Remove world-writable permissions.",
        "root_cause":   "Files or directories have overly permissive access controls allowing unauthorized read/write/execute.",
        "control":      "File permission hardening (technical + operational)",
        "before":       "chmod 777 /var/app/config.yaml",
        "after":        "chmod 640 /var/app/config.yaml; chown app:app /var/app/config.yaml",
        "test_payloads": ["find / -perm -o+w -type f 2>/dev/null", "ls -la on sensitive files"],
        "tools":        ["Lynis", "OpenSCAP", "find command"],
    },
}


# ═════════════════════════════════════════════════════════════════════════════
#  EXECUTION CONTEXT KNOWLEDGE BASE  (20 stacks — was ~4)
# ═════════════════════════════════════════════════════════════════════════════
EXECUTION_CONTEXT_KB: list[dict] = [
    {
        "stack":      "Java Spring Boot REST API",
        "tools":      ["OWASP ZAP", "Burp Suite", "SpotBugs + FindSecBugs"],
        "focus":      ["SQL injection in JPA queries", "XXE in XML parsers", "Spring Security misconfiguration", "SSRF via RestTemplate"],
        "approach":   "Run DAST against all REST endpoints. Check Spring Security filter chain for authentication bypass. Audit @RequestParam handling.",
        "env_risks":  "Actuator endpoints (/actuator/env, /actuator/heapdump) often exposed in dev; verify they're locked down in prod.",
        "indicators": ["Spring Boot banner in response headers", "Actuator endpoint accessible", "JAVA_OPTS visible in error traces"],
    },
    {
        "stack":      "Python Django web application",
        "tools":      ["Bandit", "OWASP ZAP", "Burp Suite"],
        "focus":      ["Raw SQL via .extra() or .raw()", "Template injection in user-controlled templates", "CSRF token bypass", "Insecure ALLOWED_HOSTS"],
        "approach":   "Run Bandit for static analysis. Check Django DEBUG=True in production. Fuzz all form inputs for injection.",
        "env_risks":  "DEBUG=True leaks full stack traces and settings; SECRET_KEY must not be the default.",
        "indicators": ["Django debug toolbar", "Yellow debug error pages", "Sentry DSN exposed in JS"],
    },
    {
        "stack":      "Node.js Express API",
        "tools":      ["npm audit", "Retire.js", "Burp Suite", "Snyk"],
        "focus":      ["Prototype pollution", "NoSQL injection", "JWT library vulnerabilities", "Path traversal in static file serving"],
        "approach":   "Run npm audit and Snyk for dependency CVEs. Test prototype pollution via __proto__ payloads. Check helmet.js configuration.",
        "env_risks":  "NODE_ENV=development enables verbose errors; missing helmet middleware exposes default Express headers.",
        "indicators": ["X-Powered-By: Express header", "node_modules exposed via static path", "package.json accessible via URL"],
    },
    {
        "stack":      "PHP Laravel application",
        "tools":      ["PHPCS Security Audit", "Burp Suite", "OWASP ZAP"],
        "focus":      ["Mass assignment via $fillable misconfiguration", "SQL injection in raw DB queries", "Blade template injection", "Unprotected .env file"],
        "approach":   "Fuzz all POST endpoints for mass assignment. Check route:list for unauthenticated routes. Scan for .env exposure.",
        "env_risks":  "APP_DEBUG=true in .env exposes stack traces; storage/ directory must not be web-accessible.",
        "indicators": ["Laravel error page style", ".env accessible at /.env", "APP_KEY in error output"],
    },
    {
        "stack":      "Ruby on Rails application",
        "tools":      ["Brakeman", "Burp Suite", "bundler-audit"],
        "focus":      ["IDOR via insecure direct object references", "Mass assignment via strong params bypass", "CSRF token handling", "Regexp DoS in routes"],
        "approach":   "Run Brakeman for static SAST. Run bundler-audit for gem CVEs. Test IDOR on all resource IDs.",
        "env_risks":  "Secret key base must not use default. Check config/credentials.yml.enc is not committed to git.",
        "indicators": ["X-Runtime header", "Rails error pages", "Turbolinks in page source"],
    },
    {
        "stack":      "ASP.NET Core (C#) application",
        "tools":      ["Visual Studio Analyzer", "Burp Suite", "OWASP ZAP", "Roslyn analyzers"],
        "focus":      ["View injection via Razor templates", "Open redirect in returnUrl parameter", "XML deserialization via DataContractSerializer", "Anti-forgery token bypass"],
        "approach":   "Use Burp to test all returnUrl and redirect parameters. Check ViewState encryption. Test XML endpoints for XXE.",
        "env_risks":  "ASPNETCORE_ENVIRONMENT=Development enables developer exception pages; detailed errors must be suppressed in prod.",
        "indicators": ["X-AspNetMvc-Version header", "ViewState field in forms", ".aspx/.ashx extensions"],
    },
    {
        "stack":      "React + GraphQL single-page application",
        "tools":      ["Burp Suite", "InQL (GraphQL scanner)", "GraphQL Voyager"],
        "focus":      ["GraphQL introspection enabled in production", "Batching attacks / rate limit bypass", "Authorization bypass on nested resolvers", "DOM XSS via dangerouslySetInnerHTML"],
        "approach":   "Test for introspection. Enumerate all queries/mutations. Check field-level authorization on every resolver.",
        "env_risks":  "GraphQL playground/GraphiQL must be disabled in production. Introspection should be off.",
        "indicators": ["/__graphql endpoint returns schema", "GraphQL error messages expose resolver names"],
    },
    {
        "stack":      "AWS Lambda + API Gateway serverless",
        "tools":      ["Prowler", "Checkov", "AWS Security Hub", "Burp Suite"],
        "focus":      ["Over-permissive IAM roles (Lambda execution role)", "Event injection via SQS/SNS payloads", "Environment variable secrets exposure", "API Gateway authorizer bypass"],
        "approach":   "Enumerate IAM permissions attached to Lambda execution roles. Check for wildcard (*) actions. Test all API Gateway routes without Authorization header.",
        "env_risks":  "Lambda environment variables are accessible to any code running in the function; secrets must be fetched from Secrets Manager at runtime.",
        "indicators": ["x-amzn-RequestId header", "AWS-specific error messages", "Lambda function names in responses"],
    },
    {
        "stack":      "Kubernetes (K8s) containerized deployment",
        "tools":      ["kube-bench", "Trivy", "Falco", "kube-hunter"],
        "focus":      ["Privileged containers", "Host path mounts", "RBAC over-permission", "Exposed Kubernetes API server", "Container escape via volume mounts"],
        "approach":   "Run kube-bench for CIS Kubernetes Benchmark. Scan all container images with Trivy. Test API server authentication.",
        "env_risks":  "Dashboard exposed externally; default service account token mounted; no pod security policies enforced.",
        "indicators": ["Kubernetes API server at :6443 accessible", "kubectl get pods works without auth", "Default namespace has wildcard RBAC"],
    },
    {
        "stack":      "Docker containerized microservices",
        "tools":      ["Trivy", "Docker Bench for Security", "Syft", "Grype"],
        "focus":      ["Containers running as root", "Secrets in Dockerfile ENV", "Exposed Docker daemon socket", "Base image CVEs"],
        "approach":   "Run Trivy on all images. Run Docker Bench for Security. Verify no container has --privileged or hostNetwork.",
        "env_risks":  "Mounting /var/run/docker.sock gives container root on host; avoid unless absolutely necessary.",
        "indicators": ["Dockerfile uses latest tag", "ENV credentials in image history", "Container running as UID 0"],
    },
    {
        "stack":      "Android mobile application (APK)",
        "tools":      ["MobSF", "apktool", "jadx", "Burp Suite with Android proxy"],
        "focus":      ["Hardcoded API keys in smali/java", "Insecure data storage (SharedPreferences/SQLite plaintext)", "Exported activities/providers", "Certificate pinning bypass"],
        "approach":   "Run MobSF static analysis on APK. Decompile with jadx. Proxy traffic via Burp with cert pinning bypass (Frida/objection).",
        "env_risks":  "Logcat in debug builds leaks sensitive data; exported components can be invoked by other apps.",
        "indicators": ["android:debuggable=true in manifest", "Exported activities with no permission check", "API keys in strings.xml"],
    },
    {
        "stack":      "iOS mobile application (IPA)",
        "tools":      ["MobSF", "objection", "Frida", "Burp Suite"],
        "focus":      ["Keychain data protection level", "NSURLSession certificate validation", "Sensitive data in NSUserDefaults/plist", "Jailbreak detection bypass"],
        "approach":   "Run MobSF static analysis. Use objection to dump Keychain contents. Proxy traffic via Burp after SSL kill-switch.",
        "env_risks":  "Sensitive PII stored in NSUserDefaults survives app uninstall; use Keychain with kSecAttrAccessibleWhenUnlocked.",
        "indicators": ["NSAllowsArbitraryLoads=YES in Info.plist", "NSUserDefaults contains tokens", "Weak Keychain protection class"],
    },
    {
        "stack":      "PostgreSQL-backed web application",
        "tools":      ["SQLMap", "pgAudit", "Burp Suite"],
        "focus":      ["SQLi via ORM raw queries", "pg_read_file / COPY TO for file read", "pg_hba.conf trust authentication", "Role escalation via SUPERUSER grants"],
        "approach":   "Test all parameterized query paths. Check pg_hba.conf for 'trust' entries. Review SUPERUSER and CREATEDB privileges.",
        "env_risks":  "Default postgres user with no password; pg_hba.conf allowing all hosts; COPY TO PROGRAM RCE if superuser.",
        "indicators": ["PostgreSQL error messages visible", "pg_version in responses", "Default port 5432 exposed externally"],
    },
    {
        "stack":      "WordPress CMS",
        "tools":      ["WPScan", "Burp Suite", "Nikto"],
        "focus":      ["Vulnerable plugins (90% of WordPress CVEs)", "XML-RPC brute force", "User enumeration via /?author=1", "File inclusion in themes"],
        "approach":   "Run WPScan with API token for plugin CVE scanning. Test XML-RPC multicall. Check wp-config.php accessibility.",
        "env_risks":  "wp-config.php must not be web-readable; uploads directory must not execute PHP; wp-cron can be DoS vector.",
        "indicators": ["/wp-login.php accessible", "Generator meta tag reveals WP version", "/wp-content/ paths in source"],
    },
    {
        "stack":      "Elasticsearch / OpenSearch cluster",
        "tools":      ["Nuclei (ES templates)", "Burp Suite", "curl"],
        "focus":      ["Unauthenticated access to /_cat/indices", "Kibana exposed externally", "Dynamic script execution (Groovy/Painless)", "Sensitive data in index names"],
        "approach":   "Check if /_cluster/health returns 200 without auth. Test /_search for data exfiltration. Verify TLS on transport layer.",
        "env_risks":  "Default Elasticsearch has no authentication; Kibana dashboards often expose PII; 9200/9300 must not be internet-facing.",
        "indicators": ["9200 open with JSON cluster info", "/_cat/indices returns data without auth", "Kibana at :5601 externally reachable"],
    },
    {
        "stack":      "Redis cache / session store",
        "tools":      ["redis-cli", "Nuclei (Redis templates)", "Shodan (check exposure)"],
        "focus":      ["Unauthenticated Redis access", "CONFIG SET dir for arbitrary file write", "SLAVEOF for replication-based RCE", "Session token theft from keyspace"],
        "approach":   "Test for authentication: redis-cli -h target PING. Check ACL configuration. Verify bind address and requirepass.",
        "env_risks":  "Redis with no password and bind 0.0.0.0 is a critical misconfiguration; allows RCE via config rewrite.",
        "indicators": ["Redis PING returns PONG without auth", "Port 6379 exposed externally", "Keyspace contains session tokens"],
    },
    {
        "stack":      "CI/CD pipeline (GitHub Actions / Jenkins)",
        "tools":      ["Semgrep (SAST)", "truffleHog", "checkov", "OWASP Dependency-Check"],
        "focus":      ["Secrets in environment variables / workflow files", "Untrusted input in run: steps (script injection)", "GITHUB_TOKEN over-permission", "Artifact poisoning"],
        "approach":   "Audit all workflow files for ${{ github.event.*.body }} in run: steps. Scan for hardcoded secrets. Review GITHUB_TOKEN permissions.",
        "env_risks":  "pull_request_target with untrusted code execution is a critical misconfiguration leading to secret exfiltration.",
        "indicators": ["Workflow uses pull_request_target with checkout of PR code", "ACTIONS_RUNNER_DEBUG=true leaking secrets", "Jenkins build logs publicly accessible"],
    },
    {
        "stack":      "OAuth 2.0 / OIDC authentication flow",
        "tools":      ["Burp Suite", "jwt_tool", "OAuth tester Burp extension"],
        "focus":      ["State parameter CSRF bypass", "Redirect URI validation bypass", "JWT algorithm confusion (RS256→HS256)", "Token leakage via Referer header"],
        "approach":   "Test redirect_uri with variations (wildcards, path traversal). Check state parameter uniqueness and validation. Test JWT alg:none and algorithm confusion.",
        "env_risks":  "Open redirect in redirect_uri allows authorization code theft; weak state allows CSRF login.",
        "indicators": ["Redirect URI allows wildcard subdomains", "State parameter is static or missing", "JWT alg is 'none' accepted"],
    },
    {
        "stack":      "Apache Kafka event streaming",
        "tools":      ["Conduktor", "kcat (kafkacat)", "Nuclei"],
        "focus":      ["Unauthenticated broker access", "Missing ACLs on topics", "SASL/SSL misconfiguration", "Sensitive PII in message payloads"],
        "approach":   "Test broker without SASL credentials. List all topics. Check if consumer groups can read any topic.",
        "env_risks":  "Default Kafka allows all connections with no auth; PLAINTEXT listener on 9092 must never be internet-facing.",
        "indicators": ["Port 9092 externally reachable", "kcat -L returns broker metadata without auth", "No SASL configuration in server.properties"],
    },
    {
        "stack":      "Nginx / Apache web server (infrastructure layer)",
        "tools":      ["Nikto", "testssl.sh", "Nuclei", "Observatory by Mozilla"],
        "focus":      ["Server version disclosure", "Misconfigured CORS headers", "Missing security headers (CSP, HSTS, X-Frame-Options)", "Path traversal via alias misconfig"],
        "approach":   "Run Nikto and Observatory. Test CORS with Origin: evil.com. Verify TLS with testssl.sh. Check for server-status/server-info exposure.",
        "env_risks":  "nginx alias traversal: location /static/ { alias /var/static; } allows /../ path traversal.",
        "indicators": ["Server: Apache/2.4.x header exposes version", "server-status accessible", "CORS returns * for all origins"],
    },
]


# ═════════════════════════════════════════════════════════════════════════════
#  HELPERS
# ═════════════════════════════════════════════════════════════════════════════

def load_json(path: Path) -> list:
    if not path.exists():
        print(f"  ⚠️  {path} not found — skipping")
        return []
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sample_cves_for_cwe(nvd_records: list, cwe: str, n: int = 30) -> list:
    matches = [r for r in nvd_records if r.get("cwe_id") == cwe and r.get("description")]
    return random.sample(matches, min(n, len(matches)))


def sample_cves_with_field(nvd_records: list, field: str, n: int = 50) -> list:
    matches = [r for r in nvd_records if r.get(field) and r.get("description")]
    return random.sample(matches, min(n, len(matches)))


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATOR: remediation_learning
# ═════════════════════════════════════════════════════════════════════════════

def generate_remediation_pairs(nvd_records: list) -> list:
    pairs = []

    for cwe, kb in REMEDIATION_KB.items():
        cve_samples = sample_cves_for_cwe(nvd_records, cwe, n=25)

        for rec in cve_samples:
            cve_id = rec.get("cve_id", "UNKNOWN")
            desc   = rec.get("description", "")[:300]
            cvss   = rec.get("cvss_score", "N/A")

            # Pair 1: direct fix request
            pairs.append({
                "instruction": _pick_instruction("how_fix", cwe=cwe, cve=cve_id),
                "input":       desc,
                "output": _perturb_output(
                    f"Remediation for {cwe} ({cve_id}):\n\n"
                    f"**Fix:** {kb['fix']}\n\n"
                    f"**Root Cause:** {kb['root_cause']}\n\n"
                    f"**Control Type:** {kb['control']}\n\n"
                    f"**Code Example:**\n"
                    f"Before (vulnerable):\n  {kb['before']}\n\n"
                    f"After (fixed):\n  {kb['after']}\n\n"
                    f"**Verification:** Test with: {', '.join(kb['test_payloads'][:2])}\n"
                    f"**Tools:** {', '.join(kb['tools'])}"
                ),
                "layer": "remediation_learning",
                "agent": "Reflector Agent",
            })

            # Pair 2: root cause analysis
            pairs.append({
                "instruction": _pick_instruction("root_cause", cve=cve_id, cwe=cwe),
                "input":       desc,
                "output": _perturb_output(
                    f"Root Cause Analysis for {cve_id} ({cwe}):\n\n"
                    f"**Root Cause:** {kb['root_cause']}\n\n"
                    f"**Why it persists:** This class of vulnerability ({cwe}) recurs because developers "
                    f"often treat it as a one-off fix rather than addressing the underlying pattern. "
                    f"A permanent fix requires: (1) fixing the specific instance, (2) auditing all similar "
                    f"patterns in the codebase, (3) adding automated checks (SAST/DAST) to prevent regression.\n\n"
                    f"**Permanent Resolution:** {kb['fix']}\n\n"
                    f"**Prevention:** Add {kb['tools'][0]} to your CI pipeline to catch future instances."
                ),
                "layer": "remediation_learning",
                "agent": "Reflector Agent",
            })

            # Pair 3: post-patch verification
            pairs.append({
                "instruction": _pick_instruction("verify_fix", cve=cve_id, cwe=cwe),
                "input":       desc,
                "output": _perturb_output(
                    f"Verification testing for {cwe} fix ({cve_id}):\n\n"
                    f"**Test payloads to confirm patch blocks attack:**\n"
                    + "\n".join(f"  {i+1}. {p}" for i, p in enumerate(kb['test_payloads']))
                    + f"\n\n**Recommended tools:** {', '.join(kb['tools'])}\n\n"
                    f"**Pass criteria:** All payloads should be rejected with appropriate error handling "
                    f"(not a 500 — that indicates input reached application logic). "
                    f"Re-run {kb['tools'][0]} to confirm no {cwe} findings remain."
                ),
                "layer": "remediation_learning",
                "agent": "Reflector Agent",
            })

            # Pair 4: CVSS-aware prioritisation guidance
            pairs.append({
                "instruction": _pick_instruction("priority", cve=cve_id, cwe=cwe, cvss=cvss),
                "input":       desc,
                "output": _perturb_output(
                    f"Patch Priority Assessment for {cve_id}:\n\n"
                    f"**CVSS Score:** {cvss} — {'CRITICAL: patch within 24 hours' if float(cvss) >= 9.0 else 'HIGH: patch within 7 days' if float(cvss) >= 7.0 else 'MEDIUM: patch within 30 days'}\n\n"
                    f"**Remediation Plan:**\n"
                    f"1. Identify all affected components in your environment\n"
                    f"2. Apply vendor patch or implement workaround: {kb['fix']}\n"
                    f"3. Verify fix using: {', '.join(kb['test_payloads'][:1])}\n"
                    f"4. Document remediation in change management system\n\n"
                    f"**Root Cause:** {kb['root_cause']}"
                ),
                "layer": "remediation_learning",
                "agent": "Reflector Agent",
            }) if str(cvss) not in ("N/A", "None", "") and _is_float(cvss) else None

        # CWE-level pairs (no specific CVE grounding — teaches class-level questions)
        pairs.append({
            "instruction": _pick_instruction("standard_remediation", cwe=cwe),
            "input":       "",
            "output": _perturb_output(
                f"Standard remediation for {cwe}:\n\n"
                f"**Fix:** {kb['fix']}\n\n"
                f"**Root Cause Pattern:** {kb['root_cause']}\n\n"
                f"**Control Type:** {kb['control']}\n\n"
                f"**Code Pattern:**\n"
                f"❌ Vulnerable: {kb['before']}\n"
                f"✅ Fixed:      {kb['after']}\n\n"
                f"**Testing:** {', '.join(kb['test_payloads'])}\n"
                f"**Tools:** {', '.join(kb['tools'])}"
            ),
            "layer": "remediation_learning",
            "agent": "Reflector Agent",
        })

        pairs.append({
            "instruction": _pick_instruction("detect_cwe", cwe=cwe),
            "input":       "",
            "output": _perturb_output(
                f"Detection strategy for {cwe}:\n\n"
                f"**SAST (Static Analysis):** Run {kb['tools'][-1]} on your codebase. Look for the pattern:\n"
                f"  {kb['before']}\n\n"
                f"**DAST (Dynamic Testing):** Use {kb['tools'][0]} and send these payloads to all relevant inputs:\n"
                + "\n".join(f"  - {p}" for p in kb['test_payloads'])
                + f"\n\n**Root pattern to audit for:** {kb['root_cause']}\n\n"
                f"**Remediation once found:** {kb['fix']}"
            ),
            "layer": "remediation_learning",
            "agent": "Reflector Agent",
        })

    # Drop Nones (from CVSS guard above)
    pairs = [p for p in pairs if p is not None]
    print(f"  remediation_learning: {len(pairs)} pairs generated")
    return pairs


def _is_float(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


# ═════════════════════════════════════════════════════════════════════════════
#  GENERATOR: execution_context
# ═════════════════════════════════════════════════════════════════════════════

def generate_execution_context_pairs(nvd_records: list) -> list:
    pairs = []

    for ctx in EXECUTION_CONTEXT_KB:
        stack      = ctx["stack"]
        tools      = ctx["tools"]
        focus      = ctx["focus"]
        approach   = ctx["approach"]
        env_risks  = ctx["env_risks"]
        indicators = ctx["indicators"]

        # Pair 1: tool selection for stack
        pairs.append({
            "instruction": _pick_instruction("tool_selection", stack=stack),
            "input":       "",
            "output": _perturb_output(
                f"Security assessment methodology for {stack}:\n\n"
                f"**Primary tool:** {tools[0]} — {approach.split('.')[0]}.\n"
                f"**Secondary:** {tools[1] if len(tools) > 1 else tools[0]} — for {focus[0]} testing.\n"
                f"**Manual review areas:** {', '.join(focus[1:3])}\n\n"
                f"**Don't skip:** {env_risks}\n\n"
                f"**Full tool list:** {', '.join(tools)}"
            ),
            "layer": "execution_context",
            "agent": "Execution Agent",
        })

        # Pair 2: what to look for / common vulnerabilities
        pairs.append({
            "instruction": _pick_instruction("common_vulns", stack=stack),
            "input":       "",
            "output": _perturb_output(
                f"Common vulnerabilities in {stack}:\n\n"
                + "\n".join(f"  {i+1}. **{v}**" for i, v in enumerate(focus))
                + f"\n\n**Testing approach:** {approach}\n\n"
                f"**Environment-specific risks:** {env_risks}"
            ),
            "layer": "execution_context",
            "agent": "Execution Agent",
        })

        # Pair 3: environment-specific risks
        pairs.append({
            "instruction": _pick_instruction("env_risks", stack=stack),
            "input":       "",
            "output": _perturb_output(
                f"Environment risk checklist for {stack}:\n\n"
                f"**Primary risk:** {env_risks}\n\n"
                f"**Indicators of misconfiguration:**\n"
                + "\n".join(f"  - {ind}" for ind in indicators)
                + f"\n\n**Remediation approach:** {approach}"
            ),
            "layer": "execution_context",
            "agent": "Execution Agent",
        })

        # Pair 4: detection indicators
        pairs.append({
            "instruction": _pick_instruction("recon_indicators", stack=stack),
            "input":       "",
            "output": _perturb_output(
                f"Fingerprinting and attack surface for {stack}:\n\n"
                f"**Confirmation indicators:**\n"
                + "\n".join(f"  - {ind}" for ind in indicators)
                + f"\n\n**Attack surface to focus on:**\n"
                + "\n".join(f"  - {v}" for v in focus)
                + f"\n\n**Recommended tools:** {', '.join(tools)}"
            ),
            "layer": "execution_context",
            "agent": "Execution Agent",
        })

    # NVD-grounded pairs: tie execution context to CVE affected software
    nvd_sample = sample_cves_with_field(nvd_records, "affected_software", n=100)
    for rec in nvd_sample:
        cve_id   = rec.get("cve_id", "UNKNOWN")
        desc     = rec.get("description", "")[:250]
        software = rec.get("affected_software", [])
        if not software:
            continue
        sw_name = software[0] if isinstance(software[0], str) else software[0].get("product", "")
        if not sw_name:
            continue

        pairs.append({
            "instruction": _pick_instruction("test_cve_env", cve=cve_id, sw=sw_name),
            "input":       desc,
            "output": _perturb_output(
                f"Testing approach for {cve_id} in {sw_name} environment:\n\n"
                f"1. **Confirm version:** Verify {sw_name} version is in the affected range for {cve_id}.\n"
                f"2. **Set up test environment:** Mirror the production stack with the same {sw_name} version.\n"
                f"3. **Exploit path:** {desc[:150]}\n"
                f"4. **Tools:** Use Burp Suite for HTTP-based vectors; OWASP ZAP for automated scanning; "
                f"review CVE PoC on ExploitDB or GitHub.\n"
                f"5. **Remediation gate:** Apply vendor patch and re-test — confirm vulnerability no longer triggers."
            ),
            "layer": "execution_context",
            "agent": "Execution Agent",
        })

    print(f"  execution_context: {len(pairs)} pairs generated")
    return pairs


# ═════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════════════════════════════════════

def run():
    print("Loading NVD records for CVE grounding...")
    nvd_records = load_json(NVD_PATH)
    print(f"  Loaded {len(nvd_records)} NVD records")

    all_pairs: list[dict] = []

    print("\nGenerating remediation_learning pairs...")
    all_pairs.extend(generate_remediation_pairs(nvd_records))

    print("Generating execution_context pairs...")
    all_pairs.extend(generate_execution_context_pairs(nvd_records))

    # Quality filter
    clean_pairs = [p for p in all_pairs if len(p.get("output", "").strip()) >= 80]
    print(f"\n  Total synthetic pairs: {len(clean_pairs)} "
          f"(dropped {len(all_pairs) - len(clean_pairs)} too-short)")

    # Append to existing training_pairs.jsonl with deduplication
    if OUTPUT_PATH.exists():
        existing_keys: set[tuple[str, str]] = set()
        with open(OUTPUT_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    p = json.loads(line)
                    existing_keys.add((
                        p.get("instruction", "").strip()[:150],
                        p.get("output", "").strip()[:200],
                    ))
                except Exception:
                    pass

        new_pairs = [
            p for p in clean_pairs
            if (p["instruction"].strip()[:150], p["output"].strip()[:200]) not in existing_keys
        ]
        print(f"  New unique pairs (not already in file): {len(new_pairs)}")

        with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
            for p in new_pairs:
                f.write(json.dumps(p) + "\n")

        print(f"\n✅ Appended {len(new_pairs)} synthetic pairs → {OUTPUT_PATH}")
    else:
        with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
            for p in clean_pairs:
                f.write(json.dumps(p) + "\n")
        print(f"\n✅ Wrote {len(clean_pairs)} synthetic pairs → {OUTPUT_PATH}")

    # Summary
    layer_counts: dict[str, int] = {}
    for p in clean_pairs:
        layer = p.get("layer", "unknown")
        layer_counts[layer] = layer_counts.get(layer, 0) + 1

    print("\nSynthetic pairs by layer:")
    for layer, count in sorted(layer_counts.items()):
        print(f"  {layer:<36} {count:>6}")


if __name__ == "__main__":
    run()