"""
stack_profiles.py
─────────────────────────────────────────────────────────────────────────────
Technology stack profiles used by build_cooccurrence_v2.py and
generate_cooccurrence_pairs.py.

Each profile defines:
  - indicators:       keywords that identify this stack in a finding
  - high_confidence:  CVEs almost always present when stack is confirmed
  - conditional:      CVEs present only when sub-condition is met
  - negative_rules:   conditions under which certain CVEs are ABSENT
  - remediation_ties: groups of CVEs that share a single fix (patch one = patch all)
  - attack_chains:    ordered lists representing "step 1 → step 2 → step 3" attack paths

Auto-generated profiles (from KEV clusters + NVD product co-occurrence) are
appended at module load time via auto_generate_stack_profiles(). These carry
a `"_auto_generated": True` flag for downstream filtering / human review.
"""

import json
import logging
import re
from collections import defaultdict
from itertools import combinations
from pathlib import Path

log = logging.getLogger(__name__)

STACK_PROFILES = {

    # ──────────────────────────────────────────────────────────────────────
    # Java Enterprise
    # ──────────────────────────────────────────────────────────────────────

    "java_log4j": {
        "display_name": "Apache Log4j (Java Logging)",
        "indicators":   ["log4j", "log4j2", "log4j-core"],
        "version_field": "log4j-core",
        "high_confidence": [
            {"cve": "CVE-2021-44228", "reason": "Log4Shell — JNDI lookup via any logged string"},
            {"cve": "CVE-2021-45046", "reason": "Log4Shell bypass — thread context patterns"},
            {"cve": "CVE-2021-45105", "reason": "DoS via infinite recursion in lookup"},
        ],
        "conditional": {
            "if_version_lt_2_17_1": [
                {"cve": "CVE-2021-44832", "reason": "Attacker-controlled JDBC config RCE"},
            ],
            "if_version_lt_2_3_1_java8": [
                {"cve": "CVE-2021-44228", "reason": "Java 8 older patch incomplete"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "log4j_version >= 2.17.1 AND formatMsgNoLookups=true",
                "absent_cves": ["CVE-2021-44228", "CVE-2021-45046", "CVE-2021-45105"],
                "still_assess": ["CVE-2021-44832"],
                "reason":      "JNDI lookups disabled — core attack vector removed",
            },
            {
                "condition":   "log4j_version >= 2.17.1",
                "absent_cves": ["CVE-2021-44228", "CVE-2021-45046"],
                "reason":      "Official patch addresses JNDI injection root cause",
            },
        ],
        "remediation_ties": [
            {
                "fix":  "Upgrade log4j-core to ≥ 2.17.1",
                "cves": ["CVE-2021-44228", "CVE-2021-45046", "CVE-2021-45105"],
            }
        ],
        "attack_chains": [
            ["CVE-2021-44228", "CVE-2021-44832"],  # Initial access → config control
        ],
        "independent_assess": [
            "CVE-2022-22965",  # Spring4Shell — different component
            "CVE-2022-21449",  # Psychic Signatures — JDK level
        ],
    },

    "java_spring": {
        "display_name": "Spring Framework (Java)",
        "indicators":   ["spring", "spring-boot", "spring-framework", "springmvc"],
        "high_confidence": [
            {"cve": "CVE-2022-22965", "reason": "Spring4Shell — RCE via data binding on JDK9+"},
        ],
        "conditional": {
            "if_spring_cloud_function": [
                {"cve": "CVE-2022-22963", "reason": "SpEL injection via routing expression"},
            ],
            "if_spring_security_lt_5_6_5": [
                {"cve": "CVE-2022-22978", "reason": "Auth bypass via regex in Spring Security"},
            ],
            "if_spring_data_lt_2_6_4": [
                {"cve": "CVE-2022-22980", "reason": "SpEL injection in Spring Data MongoDB"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "JDK_version < 9 AND spring_version >= 5.3.18",
                "absent_cves": ["CVE-2022-22965"],
                "reason":      "Spring4Shell requires JDK9+ class loader module access",
            },
            {
                "condition":   "spring_version >= 5.3.18",
                "absent_cves": ["CVE-2022-22965"],
                "reason":      "Patched spring-beans version",
            },
        ],
        "remediation_ties": [
            {
                "fix":  "Upgrade spring-framework to ≥ 5.3.18 / 5.2.20",
                "cves": ["CVE-2022-22965"],
            }
        ],
        "attack_chains": [
            ["CVE-2022-22965", "CVE-2022-22963"],
        ],
    },

    "java_struts": {
        "display_name": "Apache Struts 2",
        "indicators":   ["struts2", "struts-2", "struts", "ognl"],
        "high_confidence": [
            {"cve": "CVE-2017-5638",  "reason": "Content-Type OGNL injection — Equifax breach"},
            {"cve": "CVE-2018-11776", "reason": "Namespace OGNL injection — no param required"},
            {"cve": "CVE-2019-0230",  "reason": "Forced double OGNL evaluation"},
        ],
        "conditional": {
            "if_struts_lt_2_3_35": [
                {"cve": "CVE-2017-9805",  "reason": "REST plugin XStream deserialization"},
                {"cve": "CVE-2017-12611", "reason": "Freemarker tag OGNL injection"},
            ],
            "if_struts_lt_2_5_22": [
                {"cve": "CVE-2019-0230", "reason": "Double OGNL evaluation in tag attributes"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "struts_version >= 2.5.30",
                "absent_cves": ["CVE-2017-5638", "CVE-2018-11776", "CVE-2019-0230"],
                "reason":      "All known OGNL injection vectors patched in 2.5.30+",
            },
        ],
        "remediation_ties": [
            {"fix": "Upgrade Struts2 to ≥ 2.5.30", "cves": ["CVE-2017-5638", "CVE-2018-11776", "CVE-2019-0230"]},
        ],
        "attack_chains": [
            ["CVE-2017-5638", "CVE-2017-9805"],
        ],
    },

    "java_weblogic": {
        "display_name": "Oracle WebLogic Server",
        "indicators":   ["weblogic", "wls", "t3://"],
        "high_confidence": [
            {"cve": "CVE-2019-2725",  "reason": "Deserialization via _async servlets — no auth"},
            {"cve": "CVE-2020-14882", "reason": "Auth bypass + RCE in console component"},
            {"cve": "CVE-2021-2109",  "reason": "JNDI injection via admin console"},
            {"cve": "CVE-2023-21839", "reason": "IIOP/T3 deserialization — no auth required"},
        ],
        "conditional": {
            "if_t3_port_open": [
                {"cve": "CVE-2018-2628",  "reason": "T3 protocol deserialization"},
                {"cve": "CVE-2018-2893",  "reason": "T3 protocol bypass after 2628 patch"},
            ],
            "if_iiop_enabled": [
                {"cve": "CVE-2023-21839", "reason": "IIOP deserialization — separate attack surface"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "T3_protocol_blocked_at_firewall",
                "absent_cves": ["CVE-2018-2628", "CVE-2018-2893"],
                "still_assess": ["CVE-2020-14882"],
                "reason":      "T3 CVEs require network access to T3 port (7001/7002)",
            },
        ],
        "remediation_ties": [
            {"fix": "Apply Oracle CPU October 2023", "cves": ["CVE-2023-21839"]},
            {"fix": "Apply Oracle CPU October 2020", "cves": ["CVE-2020-14882"]},
        ],
        "attack_chains": [
            ["CVE-2020-14882", "CVE-2021-2109"],
        ],
    },

    "java_shiro": {
        "display_name": "Apache Shiro (Java auth framework)",
        "indicators":   ["shiro", "rememberme", "apache-shiro"],
        "high_confidence": [
            {"cve": "CVE-2016-4437", "reason": "RememberMe cookie AES deserialization"},
            {"cve": "CVE-2019-12422", "reason": "RememberMe padding oracle"},
            {"cve": "CVE-2020-1957",  "reason": "Auth bypass via path traversal"},
        ],
        "conditional": {
            "if_shiro_lt_1_7_1": [
                {"cve": "CVE-2020-17523", "reason": "Auth bypass via empty string URL"},
                {"cve": "CVE-2020-13933", "reason": "Auth bypass special character"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "shiro_version >= 1.10.0 AND custom_key_configured",
                "absent_cves": ["CVE-2016-4437"],
                "reason":      "1.10.0 defaults to GCM; custom key removes default-key attack vector",
            },
        ],
        "attack_chains": [
            ["CVE-2016-4437", "CVE-2020-1957"],  # Deserialization → auth bypass pivot
        ],
    },

    "java_jenkins": {
        "display_name": "Jenkins CI/CD",
        "indicators":   ["jenkins", "jenkins-ci", "hudson"],
        "high_confidence": [
            {"cve": "CVE-2018-1000861", "reason": "Stapler routing bypass → RCE"},
            {"cve": "CVE-2019-1003000", "reason": "Script security sandbox bypass"},
            {"cve": "CVE-2024-23897",   "reason": "Arbitrary file read via args4j"},
        ],
        "conditional": {
            "if_script_console_enabled": [
                {"cve": "CVE-2019-1003000", "reason": "Groovy sandbox bypass via Pipeline"},
            ],
            "if_jenkins_lt_2_441": [
                {"cve": "CVE-2024-23897", "reason": "CLI file read — critical in 2024 wave"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "jenkins_version >= 2.441 AND LTS_version >= 2.426.3",
                "absent_cves": ["CVE-2024-23897"],
                "reason":      "Patched in Jenkins 2.441 — args4j expansion disabled",
            },
        ],
        "attack_chains": [
            ["CVE-2024-23897", "CVE-2019-1003000"],
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # PHP / CMS
    # ──────────────────────────────────────────────────────────────────────

    "php_drupal": {
        "display_name": "Drupal CMS (PHP)",
        "indicators":   ["drupal", "drupal-core", "drupal8", "drupal7"],
        "high_confidence": [
            {"cve": "CVE-2018-7600", "reason": "Drupalgeddon2 — remote code execution"},
            {"cve": "CVE-2018-7602", "reason": "Drupalgeddon3 — authenticated follow-on RCE"},
        ],
        "conditional": {
            "if_drupal_7": [
                {"cve": "CVE-2014-3704", "reason": "Drupageddon1 — SQLi via DB layer"},
            ],
            "if_drupal_8_lt_8_5": [
                {"cve": "CVE-2018-7600", "reason": "Unpatched Drupal 8.x < 8.5.1"},
            ],
            "if_image_module_enabled": [
                {"cve": "CVE-2019-6339", "reason": "ImageMagick RCE via phar deserialization"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "drupal_version >= 9.5.3 OR drupal_version >= 10.0.3",
                "absent_cves": ["CVE-2018-7600", "CVE-2018-7602"],
                "reason":      "Drupalgeddon patches applied in 8.5.1/9.x+",
            },
        ],
        "remediation_ties": [
            {"fix": "Drupal security update SA-CORE-2018-002", "cves": ["CVE-2018-7600"]},
            {"fix": "Drupal security update SA-CORE-2018-004", "cves": ["CVE-2018-7602"]},
        ],
        "attack_chains": [
            ["CVE-2014-3704", "CVE-2018-7600", "CVE-2018-7602"],
        ],
    },

    "php_phpunit": {
        "display_name": "PHPUnit (Dev/Testing dependency)",
        "indicators":   ["phpunit", "vendor/phpunit"],
        "high_confidence": [
            {"cve": "CVE-2017-9841", "reason": "eval() injection via /vendor/phpunit in production"},
        ],
        "negative_rules": [
            {
                "condition":   "vendor_directory_not_web_accessible",
                "absent_cves": ["CVE-2017-9841"],
                "reason":      "Exploit requires HTTP access to /vendor/ — blocked by webroot config",
            },
            {
                "condition":   "phpunit_not_installed_in_production",
                "absent_cves": ["CVE-2017-9841"],
                "reason":      "Dev dependency absent from production deployment",
            },
        ],
        "attack_chains": [],
    },

    "php_laravel": {
        "display_name": "Laravel PHP Framework",
        "indicators":   ["laravel", "artisan", ".env laravel"],
        "high_confidence": [
            {"cve": "CVE-2021-3129", "reason": "Debug mode RCE via Ignition facade"},
        ],
        "conditional": {
            "if_debug_mode_on": [
                {"cve": "CVE-2021-3129", "reason": "APP_DEBUG=true exposes Ignition endpoint"},
            ],
            "if_env_file_exposed": [
                {"cve": "CVE-2017-16894", "reason": ".env disclosure — credential exposure"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "APP_DEBUG=false AND ignition_version >= 2.5.2",
                "absent_cves": ["CVE-2021-3129"],
                "reason":      "Debug mode off — Ignition endpoint not exposed; patched version",
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Microsoft / Windows
    # ──────────────────────────────────────────────────────────────────────

    "microsoft_exchange": {
        "display_name": "Microsoft Exchange Server",
        "indicators":   ["exchange", "owa", "autodiscover", "msexchange"],
        "high_confidence": [
            {"cve": "CVE-2021-26855", "reason": "ProxyLogon — SSRF auth bypass"},
            {"cve": "CVE-2021-26857", "reason": "ProxyLogon — deserialization after bypass"},
            {"cve": "CVE-2021-26858", "reason": "ProxyLogon — arbitrary file write post-auth"},
            {"cve": "CVE-2021-27065", "reason": "ProxyLogon — arbitrary file write post-auth"},
        ],
        "conditional": {
            "if_exchange_2016_or_2019": [
                {"cve": "CVE-2021-34473", "reason": "ProxyShell — path confusion bypass"},
                {"cve": "CVE-2021-34523", "reason": "ProxyShell — elevation of privilege"},
                {"cve": "CVE-2021-31207", "reason": "ProxyShell — mailbox import RCE"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "exchange_version >= 15.2.986.15",
                "absent_cves": ["CVE-2021-26855", "CVE-2021-26857"],
                "reason":      "ProxyLogon patch applied — March 2021 SU or later",
            },
        ],
        "remediation_ties": [
            {
                "fix":  "March 2021 Exchange Security Update",
                "cves": ["CVE-2021-26855", "CVE-2021-26857", "CVE-2021-26858", "CVE-2021-27065"],
            },
            {
                "fix":  "May 2021 Exchange Security Update (ProxyShell)",
                "cves": ["CVE-2021-34473", "CVE-2021-34523", "CVE-2021-31207"],
            },
        ],
        "attack_chains": [
            # ProxyLogon chain
            ["CVE-2021-26855", "CVE-2021-26857", "CVE-2021-26858"],
            # ProxyShell chain
            ["CVE-2021-34473", "CVE-2021-34523", "CVE-2021-31207"],
        ],
    },

    "microsoft_windows_ad": {
        "display_name": "Windows Active Directory / Domain Services",
        "indicators":   ["active directory", "kerberos", "ldap", "domain controller", "ad ds"],
        "high_confidence": [
            {"cve": "CVE-2020-1472",  "reason": "Zerologon — instant DC compromise via Netlogon"},
            {"cve": "CVE-2021-42278", "reason": "sAMAccountName spoofing for privilege escalation"},
            {"cve": "CVE-2021-42287", "reason": "noPac — combined with 42278 for DA"},
        ],
        "conditional": {
            "if_smb_exposed": [
                {"cve": "CVE-2017-0144", "reason": "EternalBlue — SMBv1 RCE (WannaCry vector)"},
                {"cve": "CVE-2020-0796", "reason": "SMBGhost — SMBv3 compression buffer overflow"},
            ],
            "if_print_spooler_running": [
                {"cve": "CVE-2021-34527", "reason": "PrintNightmare — print spooler RCE"},
                {"cve": "CVE-2021-1675",  "reason": "PrintNightmare variant — LPE + RCE"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "SMBv1_disabled AND patch_MS17-010_applied",
                "absent_cves": ["CVE-2017-0144"],
                "reason":      "EternalBlue requires SMBv1; disabled + patched = not exploitable",
            },
            {
                "condition":   "print_spooler_service_disabled",
                "absent_cves": ["CVE-2021-34527", "CVE-2021-1675"],
                "reason":      "PrintNightmare requires Spooler service running",
            },
            {
                "condition":   "netlogon_secure_channel_enforced_via_gpo",
                "absent_cves": ["CVE-2020-1472"],
                "reason":      "Zerologon blocked by enforcement mode (FullSecureChannelProtection=1)",
            },
        ],
        "remediation_ties": [
            {"fix": "KB5008380 + enforcement mode GPO", "cves": ["CVE-2021-42278", "CVE-2021-42287"]},
            {"fix": "Disable print spooler on DCs",      "cves": ["CVE-2021-34527", "CVE-2021-1675"]},
        ],
        "attack_chains": [
            ["CVE-2020-1472", "CVE-2021-42278", "CVE-2021-42287"],  # Zerologon → noPac
            ["CVE-2017-0144", "CVE-2020-1472"],                      # EternalBlue → Zerologon
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Network / Infra
    # ──────────────────────────────────────────────────────────────────────

    "network_fortinet": {
        "display_name": "Fortinet (FortiGate / FortiOS)",
        "indicators":   ["fortinet", "fortigate", "fortios", "forticlient"],
        "high_confidence": [
            {"cve": "CVE-2018-13379", "reason": "FortiOS SSL VPN path traversal — credential read"},
            {"cve": "CVE-2019-11510", "reason": "Pulse Secure (often paired with Fortinet infra)"},
            {"cve": "CVE-2022-40684", "reason": "FortiOS/FortiProxy auth bypass — config write"},
            {"cve": "CVE-2023-27997", "reason": "FortiOS SSL VPN heap overflow pre-auth RCE"},
        ],
        "conditional": {
            "if_fortios_lt_7_2_5": [
                {"cve": "CVE-2023-27997", "reason": "XORtigate — unpatched SSL VPN heap overflow"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "ssl_vpn_webmode_disabled",
                "absent_cves": ["CVE-2018-13379", "CVE-2023-27997"],
                "reason":      "SSL VPN web mode disabled — HTTP endpoint not exposed",
            },
            {
                "condition":   "fortios_version >= 7.4.1 AND firmware_patched",
                "absent_cves": ["CVE-2023-27997", "CVE-2022-40684"],
                "reason":      "Heap overflow and auth bypass both patched in FortiOS 7.4.1+",
            },
            {
                "condition":   "admin_interface_not_internet_exposed AND trusted_ip_acl",
                "absent_cves": ["CVE-2022-40684"],
                "still_assess": ["CVE-2018-13379"],
                "reason":      "Auth bypass requires admin interface access; SSL VPN credential leak is separate surface",
            },
        ],
        "attack_chains": [
            ["CVE-2018-13379", "CVE-2022-40684"],  # Cred harvest → config takeover
        ],
    },

    "network_cisco": {
        "display_name": "Cisco IOS / IOS XE / ASA",
        "indicators":   ["cisco", "ios xe", "ios-xe", "cisco asa", "webui cisco"],
        "high_confidence": [
            {"cve": "CVE-2023-20198", "reason": "IOS XE WebUI auth bypass — creates root user"},
            {"cve": "CVE-2023-20273", "reason": "IOS XE command injection — follows 20198"},
            {"cve": "CVE-2018-0171",  "reason": "Smart Install RCE — no auth"},
        ],
        "conditional": {
            "if_web_ui_enabled": [
                {"cve": "CVE-2023-20198", "reason": "WebUI must be reachable (ip http server)"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "ip_http_server_disabled AND ip_http_secure_server_disabled",
                "absent_cves": ["CVE-2023-20198", "CVE-2023-20273"],
                "reason":      "WebUI access disabled at IOS level — attack surface removed",
            },
            {
                "condition":   "ios_xe_version >= 17.9.4a AND patched_advisory",
                "absent_cves": ["CVE-2023-20198", "CVE-2023-20273"],
                "reason":      "WebUI privesc chain fully patched in IOS XE 17.9.4a+",
            },
            {
                "condition":   "no_vstack AND smart_install_disabled",
                "absent_cves": ["CVE-2018-0171"],
                "reason":      "Smart Install RCE requires vstack director/client; disabled = not exploitable",
            },
        ],
        "remediation_ties": [
            {
                "fix":  "Apply Cisco advisory cisco-sa-iosxe-webui-privesc-j22SaA4z",
                "cves": ["CVE-2023-20198", "CVE-2023-20273"],
            },
        ],
        "attack_chains": [
            ["CVE-2023-20198", "CVE-2023-20273"],
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Middleware / Message Brokers
    # ──────────────────────────────────────────────────────────────────────

    "middleware_activemq": {
        "display_name": "Apache ActiveMQ",
        "indicators":   ["activemq", "active-mq", "activemq broker"],
        "high_confidence": [
            {"cve": "CVE-2023-46604", "reason": "ClassInfo deserialization RCE — ExceptionResponse"},
            {"cve": "CVE-2022-41678", "reason": "Jolokia/API RCE via JMX"},
            {"cve": "CVE-2016-3088",  "reason": "Fileserver upload arbitrary file write"},
        ],
        "negative_rules": [
            {
                "condition":   "activemq_version >= 5.15.16 AND version >= 5.16.7",
                "absent_cves": ["CVE-2023-46604"],
                "reason":      "CVE-2023-46604 patched in ActiveMQ 5.15.16 / 5.16.7",
            },
            {
                "condition":   "fileserver_servlet_disabled",
                "absent_cves": ["CVE-2016-3088"],
                "reason":      "Fileserver disabled in activemq.xml — upload vector removed",
            },
        ],
        "attack_chains": [
            ["CVE-2016-3088", "CVE-2022-41678"],
        ],
    },

    "middleware_redis": {
        "display_name": "Redis (in-memory data store)",
        "indicators":   ["redis", "redis-server", ":6379"],
        "high_confidence": [
            {"cve": "CVE-2022-0543",  "reason": "Lua sandbox escape on Debian-packaged Redis"},
        ],
        "conditional": {
            "if_no_auth_required": [
                {"cve": "CVE-2015-4335", "reason": "Unauthenticated eval() code execution"},
            ],
            "if_redis_lt_7_0": [
                {"cve": "CVE-2022-24736", "reason": "SRANDMEMBER crash / potential memory disclosure"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "requirepass_set AND bind_127_0_0_1",
                "absent_cves": ["CVE-2015-4335"],
                "reason":      "Auth required + loopback bind — no remote unauthenticated access",
            },
        ],
        "attack_chains": [],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Confluence / Atlassian
    # ──────────────────────────────────────────────────────────────────────

    "atlassian_confluence": {
        "display_name": "Atlassian Confluence",
        "indicators":   ["confluence", "atlassian confluence", "confluence server"],
        "high_confidence": [
            {"cve": "CVE-2022-26134", "reason": "OGNL injection — no auth required"},
            {"cve": "CVE-2021-26084", "reason": "OGNL injection — pre-auth RCE"},
            {"cve": "CVE-2023-22527", "reason": "Template injection RCE — Confluence Data Center"},
        ],
        "conditional": {
            "if_confluence_lt_7_4_17": [
                {"cve": "CVE-2021-26084", "reason": "Unpatched pre-7.4.17 — exploitable pre-auth"},
            ],
            "if_confluence_data_center": [
                {"cve": "CVE-2023-22527", "reason": "Data Center specific template injection"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "confluence_version >= 7.19.16",
                "absent_cves": ["CVE-2022-26134"],
                "reason":      "Patched in 7.19.16 LTS — OGNL injection blocked",
            },
        ],
        "attack_chains": [
            ["CVE-2022-26134", "CVE-2023-22527"],
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # DevOps / Source Control
    # ──────────────────────────────────────────────────────────────────────

    "devops_gitlab": {
        "display_name": "GitLab CE/EE",
        "indicators":   ["gitlab", "gitlab-ce", "gitlab-ee"],
        "high_confidence": [
            {"cve": "CVE-2021-22205", "reason": "ExifTool RCE via image upload — no auth"},
            {"cve": "CVE-2023-7028",  "reason": "Account takeover via password reset"},
        ],
        "conditional": {
            "if_gitlab_lt_13_10_3": [
                {"cve": "CVE-2021-22205", "reason": "Unpatched — ExifTool 7.04 bundled"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "gitlab_version >= 16.7.2",
                "absent_cves": ["CVE-2023-7028"],
                "reason":      "Password reset bypass patched in 16.7.2",
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Cloud / Container
    # ──────────────────────────────────────────────────────────────────────

    "cloud_kubernetes": {
        "display_name": "Kubernetes (container orchestration)",
        "indicators":   ["kubernetes", "kubectl", "k8s", "kubelet", "kube-apiserver"],
        "high_confidence": [
            {"cve": "CVE-2018-1002105", "reason": "API server request proxy privesc"},
            {"cve": "CVE-2019-11247",   "reason": "API server path confusion — cluster-scope access"},
            {"cve": "CVE-2022-3294",    "reason": "Node address bypasses — Kubelet auth bypass"},
        ],
        "conditional": {
            "if_anonymous_auth_enabled": [
                {"cve": "CVE-2019-11248", "reason": "/debug/pprof exposed without auth"},
            ],
            "if_etcd_exposed_no_auth": [
                {"cve": "CVE-2020-15106", "reason": "etcd raft panic — DoS"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "anonymous_auth_disabled AND RBAC_enforced",
                "absent_cves": ["CVE-2019-11248"],
                "reason":      "Debug endpoint requires auth when anonymous disabled",
            },
            {
                "condition":   "kubernetes_version >= 1.14.0 AND api_server_patched",
                "absent_cves": ["CVE-2018-1002105"],
                "reason":      "Back-end websocket hijack patched in v1.10.11 / v1.11.5 / v1.12.3+",
            },
            {
                "condition":   "PodSecurityAdmission_enforce AND no_privileged_containers",
                "absent_cves": ["CVE-2022-3294"],
                "reason":      "Node address validation exploits require privileged pod scheduling",
            },
            {
                "condition":   "etcd_tls_client_auth AND etcd_not_internet_exposed",
                "absent_cves": ["CVE-2020-15106"],
                "still_assess": ["CVE-2018-1002105"],
                "reason":      "etcd mutual TLS blocks unauthenticated raft requests; API server still applicable",
            },
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Database
    # ──────────────────────────────────────────────────────────────────────

    "db_elasticsearch": {
        "display_name": "Elasticsearch",
        "indicators":   ["elasticsearch", "kibana", ":9200", "elastic stack"],
        "high_confidence": [
            {"cve": "CVE-2014-3120", "reason": "Dynamic scripting RCE — no auth"},
            {"cve": "CVE-2015-1427", "reason": "Groovy sandbox escape RCE"},
            {"cve": "CVE-2015-3337", "reason": "Directory traversal via site plugins"},
        ],
        "conditional": {
            "if_no_xpack_security": [
                {"cve": "CVE-2015-5531", "reason": "Snapshot restore path traversal"},
            ],
            "if_elasticsearch_lt_6_8_1": [
                {"cve": "CVE-2019-7614", "reason": "Response injection via log4j logger"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "dynamic_scripting_disabled AND es_version >= 1_6_0",
                "absent_cves": ["CVE-2014-3120", "CVE-2015-1427"],
                "reason":      "Dynamic scripting disabled by default in ES 1.6+ — root cause removed",
            },
            {
                "condition":   "xpack_security_enabled AND tls_configured",
                "absent_cves": ["CVE-2015-5531"],
                "reason":      "X-Pack security layer enforces auth on snapshot API",
            },
        ],
        "attack_chains": [
            ["CVE-2014-3120", "CVE-2015-1427"],
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # AI / ML Tooling (emerging stack)
    # ──────────────────────────────────────────────────────────────────────

    "ai_comfyui_gradio": {
        "display_name": "AI Tooling (ComfyUI / Gradio / Ollama)",
        "indicators":   ["comfyui", "gradio", "ollama", "stable diffusion", "automatic1111"],
        "high_confidence": [
            {"cve": "CVE-2025-67303", "reason": "ComfyUI SSRF + path traversal"},
            {"cve": "CVE-2023-51449", "reason": "Gradio auth bypass path traversal"},
        ],
        "conditional": {
            "if_public_facing": [
                {"cve": "CVE-2025-67303", "reason": "SSRF exploitable when ComfyUI internet-exposed"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "gradio_version >= 4.11.0",
                "absent_cves": ["CVE-2023-51449"],
                "reason":      "Path traversal patched in Gradio 4.11.0",
            },
        ],
        "attack_chains": [],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Node.js / npm ecosystem
    # ──────────────────────────────────────────────────────────────────────

    "nodejs_npm": {
        "display_name": "Node.js / npm (server-side JavaScript)",
        "indicators":   ["node.js", "nodejs", "npm", "express", "package.json",
                         "yarn", "pnpm", "next.js", "nestjs", "koa"],
        "high_confidence": [
            {"cve": "CVE-2019-10744", "reason": "Lodash prototype pollution — RCE via merge/defaultsDeep"},
            {"cve": "CVE-2021-23337", "reason": "Lodash template() command injection"},
            {"cve": "CVE-2022-24999", "reason": "qs prototype pollution — Express <4.17.3"},
            {"cve": "CVE-2024-21892", "reason": "Node.js permission model bypass via code generation"},
            {"cve": "CVE-2023-32002", "reason": "Node.js policy bypass via Module._load"},
        ],
        "conditional": {
            "if_express_used": [
                {"cve": "CVE-2022-24999", "reason": "qs prototype pollution affects Express body parsing"},
                {"cve": "CVE-2024-29041", "reason": "Express open redirect via malformed URL"},
            ],
            "if_ssr_enabled": [
                {"cve": "CVE-2023-46729", "reason": "Next.js SSRF via next-server in SSR mode"},
            ],
            "if_npm_scripts_untrusted": [
                {"cve": "CVE-2024-21892", "reason": "Malicious npm package scripts exploit code generation bypass"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "lodash_version >= 4.17.21",
                "absent_cves": ["CVE-2019-10744", "CVE-2021-23337"],
                "reason":      "Prototype pollution and template injection both patched in lodash 4.17.21",
            },
            {
                "condition":   "express_version >= 4.18.2 AND qs_version >= 6.11.0",
                "absent_cves": ["CVE-2022-24999"],
                "reason":      "qs prototype pollution fixed in qs 6.11.0 / Express 4.18.2",
            },
            {
                "condition":   "node_version >= 20.11.1 AND experimental_policy_not_used",
                "absent_cves": ["CVE-2023-32002", "CVE-2024-21892"],
                "reason":      "Permission model bypass & Module._load exploits fixed in 20.11.1+",
            },
            {
                "condition":   "Object.freeze(Object.prototype) AND no_unsafe_merge",
                "absent_cves": ["CVE-2019-10744"],
                "still_assess": ["CVE-2021-23337"],
                "reason":      "Frozen prototype blocks pollution but template injection uses different path",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2019-10744", "CVE-2021-23337"],
             "fix": "Upgrade lodash to >= 4.17.21 across all workspace packages"},
        ],
        "attack_chains": [
            ["CVE-2022-24999", "CVE-2019-10744"],  # qs pollution → lodash merge chain
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Python / pip ecosystem
    # ──────────────────────────────────────────────────────────────────────

    "python_pip": {
        "display_name": "Python / pip (server-side Python)",
        "indicators":   ["python", "pip", "flask", "django", "fastapi", "requirements.txt",
                         "pyproject.toml", "poetry", "uvicorn", "gunicorn", "celery"],
        "high_confidence": [
            {"cve": "CVE-2022-42889",  "reason": "Apache Commons Text — StringSubstitutor RCE (affects Jython/polyglot)"},
            {"cve": "CVE-2021-29921",  "reason": "Python ipaddress SSRF bypass via leading zeros"},
            {"cve": "CVE-2023-37920",  "reason": "Certifi removes e-Tugra root CA — MitM risk"},
            {"cve": "CVE-2022-40897",  "reason": "Setuptools regex DoS via package URL"},
        ],
        "conditional": {
            "if_flask_used": [
                {"cve": "CVE-2023-30861", "reason": "Flask session cookie caching on proxied HTTP"},
                {"cve": "CVE-2019-1010083", "reason": "Flask Werkzeug debugger PIN RCE"},
            ],
            "if_django_used": [
                {"cve": "CVE-2023-36053", "reason": "Django EmailValidator/URLValidator ReDoS"},
                {"cve": "CVE-2024-27351", "reason": "Django Truncator ReDoS via lazy string"},
            ],
            "if_pyyaml_unsafe_load": [
                {"cve": "CVE-2020-14343", "reason": "PyYAML full_load/unsafe_load arbitrary code exec"},
            ],
            "if_jinja2_user_templates": [
                {"cve": "CVE-2024-22195", "reason": "Jinja2 XSS in xmlattr filter"},
            ],
            "if_pillow_used": [
                {"cve": "CVE-2023-44271", "reason": "Pillow DoS via decompression bomb"},
                {"cve": "CVE-2022-22817", "reason": "Pillow ImageMath.eval() arbitrary code exec"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "pyyaml_safe_load_only AND no_unsafe_load_calls",
                "absent_cves": ["CVE-2020-14343"],
                "reason":      "safe_load() restricts tag resolution — blocks arbitrary object construction",
            },
            {
                "condition":   "django_version >= 4.2.8 AND python_version >= 3.11",
                "absent_cves": ["CVE-2023-36053", "CVE-2024-27351"],
                "reason":      "ReDoS in EmailValidator and Truncator fixed in Django 4.2.8+",
            },
            {
                "condition":   "pillow_version >= 10.0.1 AND MAX_IMAGE_PIXELS_set",
                "absent_cves": ["CVE-2023-44271", "CVE-2022-22817"],
                "reason":      "Decompression bomb guarded by MAX_IMAGE_PIXELS and eval() removed in 10.0+",
            },
            {
                "condition":   "flask_debugger_disabled AND WERKZEUG_DEBUG_PIN=off",
                "absent_cves": ["CVE-2019-1010083"],
                "still_assess": ["CVE-2023-30861"],
                "reason":      "Debugger PIN RCE requires debug mode; session caching is separate issue",
            },
            {
                "condition":   "jinja2_autoescape_enabled AND sandbox_environment_used",
                "absent_cves": ["CVE-2024-22195"],
                "reason":      "Autoescaping neutralises xmlattr XSS; SandboxedEnvironment blocks SSTI",
            },
            {
                "condition":   "python_version >= 3.11.4 AND no_octal_ip_parsing",
                "absent_cves": ["CVE-2021-29921"],
                "reason":      "ipaddress module rejects leading-zero octets since 3.11.4",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2023-36053", "CVE-2024-27351"],
             "fix": "Upgrade Django to >= 4.2.8 / >= 5.0.1"},
            {"cves": ["CVE-2023-44271", "CVE-2022-22817"],
             "fix": "Upgrade Pillow to >= 10.0.1 and set Image.MAX_IMAGE_PIXELS"},
        ],
        "attack_chains": [
            ["CVE-2019-1010083", "CVE-2020-14343"],  # Werkzeug debug → PyYAML deserialization
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Docker / Container Runtime
    # ──────────────────────────────────────────────────────────────────────

    "docker_container": {
        "display_name": "Docker / containerd (container runtime)",
        "indicators":   ["docker", "containerd", "runc", "dockerfile", "docker-compose",
                         "moby", "podman", "cri-o"],
        "high_confidence": [
            {"cve": "CVE-2019-5736",  "reason": "runC container escape — overwrite host binary"},
            {"cve": "CVE-2020-15257", "reason": "containerd host networking container escape"},
            {"cve": "CVE-2024-21626", "reason": "runC Leaky Vessels — fd leak container escape"},
            {"cve": "CVE-2022-0847",  "reason": "DirtyPipe — kernel pipe privesc from container to host"},
        ],
        "conditional": {
            "if_host_network_mode": [
                {"cve": "CVE-2020-15257", "reason": "containerd-shim host networking API exposed"},
            ],
            "if_docker_socket_mounted": [
                {"cve": "CVE-2019-5736", "reason": "Mounted socket amplifies runC escape to host daemon"},
            ],
            "if_privileged_container": [
                {"cve": "CVE-2022-0847", "reason": "DirtyPipe exploitable from privileged container"},
            ],
            "if_docker_api_exposed": [
                {"cve": "CVE-2019-13139", "reason": "Docker build command injection via malicious URL"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "runc_version >= 1.1.12 AND containerd_version >= 1.7.13",
                "absent_cves": ["CVE-2024-21626", "CVE-2019-5736"],
                "reason":      "Leaky Vessels fd leak and /proc/self/exe overwrite both patched",
            },
            {
                "condition":   "rootless_mode_enabled AND no_privileged_containers",
                "absent_cves": ["CVE-2019-5736", "CVE-2022-0847"],
                "still_assess": ["CVE-2024-21626"],
                "reason":      "Rootless + no-privilege blocks host binary overwrite and kernel pipe; fd leak still applies",
            },
            {
                "condition":   "no_host_network_mode AND containerd_version >= 1.4.3",
                "absent_cves": ["CVE-2020-15257"],
                "reason":      "Abstract socket access requires host network; also patched in containerd 1.4.3",
            },
            {
                "condition":   "seccomp_default_profile AND no_cap_sys_admin",
                "absent_cves": ["CVE-2022-0847"],
                "reason":      "Default seccomp profile blocks splice() needed for DirtyPipe; CAP_SYS_ADMIN required",
            },
            {
                "condition":   "docker_socket_not_mounted AND docker_api_tls_auth",
                "absent_cves": ["CVE-2019-13139"],
                "reason":      "Build command injection requires API access; no socket mount + TLS blocks exploitation",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2024-21626", "CVE-2019-5736"],
             "fix": "Upgrade runC >= 1.1.12 and containerd >= 1.7.13"},
            {"cves": ["CVE-2020-15257"],
             "fix": "Upgrade containerd >= 1.4.3 and avoid --network=host"},
        ],
        "attack_chains": [
            ["CVE-2019-5736", "CVE-2022-0847"],   # runC escape → DirtyPipe host privesc
            ["CVE-2024-21626", "CVE-2020-15257"],  # Leaky Vessels → containerd shim escape
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Nginx Web Server
    # ──────────────────────────────────────────────────────────────────────

    "nginx_webserver": {
        "display_name": "Nginx (web server / reverse proxy)",
        "indicators":   ["nginx", "openresty", "nginx.conf", "proxy_pass",
                         "lua-nginx-module", "ingress-nginx"],
        "high_confidence": [
            {"cve": "CVE-2021-23017", "reason": "DNS resolver off-by-one heap write — RCE"},
            {"cve": "CVE-2019-20372", "reason": "HTTP request smuggling via error_page + proxy_pass"},
            {"cve": "CVE-2022-41741", "reason": "ngx_http_mp4_module buffer overread — RCE"},
            {"cve": "CVE-2024-7347",  "reason": "ngx_http_mp4_module buffer over-read in seek handling"},
        ],
        "conditional": {
            "if_resolver_configured": [
                {"cve": "CVE-2021-23017", "reason": "DNS resolver must be configured in nginx.conf"},
            ],
            "if_mp4_module_loaded": [
                {"cve": "CVE-2022-41741", "reason": "mp4 module must be loaded (ngx_http_mp4_module)"},
                {"cve": "CVE-2024-7347",  "reason": "mp4 module seek handling buffer over-read"},
            ],
            "if_ingress_nginx_used": [
                {"cve": "CVE-2023-5043", "reason": "Kubernetes ingress-nginx annotation injection"},
                {"cve": "CVE-2025-1974", "reason": "IngressNightmare — unauthenticated RCE via admission controller"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "no_resolver_directive_in_config",
                "absent_cves": ["CVE-2021-23017"],
                "reason":      "DNS resolver heap overwrite requires 'resolver' directive to be configured",
            },
            {
                "condition":   "mp4_module_not_loaded AND no_mp4_streaming",
                "absent_cves": ["CVE-2022-41741", "CVE-2024-7347"],
                "reason":      "Buffer over-read only affects ngx_http_mp4_module; absent when module not loaded",
            },
            {
                "condition":   "nginx_version >= 1.25.3 AND proxy_pass_with_uri_normalization",
                "absent_cves": ["CVE-2019-20372"],
                "reason":      "Request smuggling via error_page fixed; URI normalization prevents header confusion",
            },
            {
                "condition":   "not_ingress_nginx OR ingress_nginx_version >= 1.12.1",
                "absent_cves": ["CVE-2023-5043", "CVE-2025-1974"],
                "reason":      "IngressNightmare and annotation injection patched in ingress-nginx 1.12.1",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2022-41741", "CVE-2024-7347"],
             "fix": "Upgrade nginx >= 1.23.2 or remove mp4 module if unused"},
            {"cves": ["CVE-2023-5043", "CVE-2025-1974"],
             "fix": "Upgrade ingress-nginx >= 1.12.1 and restrict admission webhook network access"},
        ],
        "attack_chains": [
            ["CVE-2019-20372", "CVE-2021-23017"],  # request smuggling → DNS hijack chain
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # PHP / WordPress ecosystem
    # ──────────────────────────────────────────────────────────────────────

    "php_wordpress": {
        "display_name": "WordPress (PHP CMS)",
        "indicators":   ["wordpress", "wp-admin", "wp-content", "wp-login", "woocommerce",
                         "wp-config.php", "elementor", "yoast", "wpml"],
        "high_confidence": [
            {"cve": "CVE-2019-8942",  "reason": "WordPress media file upload RCE via post meta"},
            {"cve": "CVE-2022-21661", "reason": "WordPress WP_Query SQL injection"},
            {"cve": "CVE-2023-2982",  "reason": "WordPress social login auth bypass"},
        ],
        "conditional": {
            "if_xmlrpc_enabled": [
                {"cve": "CVE-2020-28032", "reason": "SSRF/credential brute via xmlrpc.php"},
            ],
            "if_file_upload_allowed": [
                {"cve": "CVE-2019-8942", "reason": "Unrestricted file upload via crafted post meta"},
            ],
            "if_woocommerce_installed": [
                {"cve": "CVE-2023-28121", "reason": "WooCommerce Payments auth bypass → admin takeover"},
            ],
            "if_elementor_installed": [
                {"cve": "CVE-2023-48777", "reason": "Elementor arbitrary file upload → RCE"},
            ],
            "if_contact_form_7_installed": [
                {"cve": "CVE-2020-35489", "reason": "CF7 unrestricted file upload via double extension"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "wordpress_version >= 6.0.3 AND auto_updates_enabled",
                "absent_cves": ["CVE-2022-21661", "CVE-2019-8942"],
                "reason":      "WP_Query SQLi patched in 6.0.3; media RCE patched much earlier (5.0.1); auto-update ensures patches",
            },
            {
                "condition":   "xmlrpc_disabled OR xmlrpc_blocked_at_waf",
                "absent_cves": ["CVE-2020-28032"],
                "reason":      "xmlrpc.php must be accessible for SSRF and brute-force exploitation",
            },
            {
                "condition":   "woocommerce_payments_not_installed OR wc_pay_version >= 5.6.2",
                "absent_cves": ["CVE-2023-28121"],
                "reason":      "Auth bypass only affects WooCommerce Payments plugin < 5.6.2",
            },
            {
                "condition":   "elementor_not_installed OR elementor_version >= 3.19.1",
                "absent_cves": ["CVE-2023-48777"],
                "reason":      "File upload RCE requires Elementor plugin < 3.19.1",
            },
            {
                "condition":   "contact_form_7_not_installed OR cf7_version >= 5.3.2",
                "absent_cves": ["CVE-2020-35489"],
                "reason":      "Double extension bypass patched in Contact Form 7 v5.3.2",
            },
            {
                "condition":   "disable_file_edit_true AND upload_mimes_restricted",
                "absent_cves": ["CVE-2019-8942"],
                "still_assess": ["CVE-2022-21661"],
                "reason":      "DISALLOW_FILE_EDIT + MIME restrict blocks file upload RCE; SQLi is separate vector",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2022-21661", "CVE-2019-8942"],
             "fix": "Enable WordPress auto-updates and upgrade core to latest"},
            {"cves": ["CVE-2023-28121"],
             "fix": "Upgrade WooCommerce Payments >= 5.6.2 or remove if unused"},
        ],
        "attack_chains": [
            ["CVE-2020-28032", "CVE-2019-8942"],   # SSRF/brute → file upload RCE
            ["CVE-2023-28121", "CVE-2023-48777"],   # WooCommerce auth bypass → Elementor file upload
        ],
    },

    # ──────────────────────────────────────────────────────────────────────
    # Apache HTTP Server
    # ──────────────────────────────────────────────────────────────────────

    "apache_httpd": {
        "display_name": "Apache HTTP Server",
        "indicators":   ["apache", "httpd", "mod_ssl", "mod_proxy", ".htaccess",
                         "apache2", "mod_cgi", "mod_rewrite"],
        "high_confidence": [
            {"cve": "CVE-2021-41773", "reason": "Path traversal + RCE via cgi-bin (Apache 2.4.49)"},
            {"cve": "CVE-2021-42013", "reason": "Incomplete fix for CVE-2021-41773 (Apache 2.4.50)"},
            {"cve": "CVE-2023-25690", "reason": "mod_proxy HTTP request smuggling via RewriteRule"},
            {"cve": "CVE-2024-38476", "reason": "mod_proxy SSRF via malicious backend response headers"},
        ],
        "conditional": {
            "if_mod_cgi_enabled": [
                {"cve": "CVE-2021-41773", "reason": "Path traversal escalates to RCE when mod_cgi active"},
            ],
            "if_mod_proxy_enabled": [
                {"cve": "CVE-2023-25690", "reason": "Request smuggling requires mod_proxy + RewriteRule"},
                {"cve": "CVE-2024-38476", "reason": "SSRF via mod_proxy response handling"},
            ],
            "if_mod_lua_enabled": [
                {"cve": "CVE-2023-45802", "reason": "HTTP/2 stream reset DoS via mod_lua"},
            ],
        },
        "negative_rules": [
            {
                "condition":   "apache_version >= 2.4.52",
                "absent_cves": ["CVE-2021-41773", "CVE-2021-42013"],
                "reason":      "Path traversal and its incomplete fix both resolved in 2.4.52+",
            },
            {
                "condition":   "mod_proxy_disabled AND no_reverse_proxy_config",
                "absent_cves": ["CVE-2023-25690", "CVE-2024-38476"],
                "reason":      "Request smuggling and SSRF require mod_proxy to be loaded and configured",
            },
            {
                "condition":   "mod_cgi_disabled AND mod_cgid_disabled",
                "absent_cves": ["CVE-2021-41773"],
                "still_assess": ["CVE-2021-42013"],
                "reason":      "Path traversal without CGI is info disclosure only; with CGI is RCE. 42013 still leaks files",
            },
            {
                "condition":   "apache_version >= 2.4.59 AND proxy_pass_with_nocanon_off",
                "absent_cves": ["CVE-2023-25690", "CVE-2024-38476"],
                "reason":      "Smuggling and SSRF fully patched in 2.4.59; nocanon disabling prevents URL confusion",
            },
        ],
        "remediation_ties": [
            {"cves": ["CVE-2021-41773", "CVE-2021-42013"],
             "fix": "Upgrade Apache HTTPD to >= 2.4.52 immediately (actively exploited)"},
            {"cves": ["CVE-2023-25690", "CVE-2024-38476"],
             "fix": "Upgrade Apache HTTPD to >= 2.4.59 and review mod_proxy RewriteRule configs"},
        ],
        "attack_chains": [
            ["CVE-2021-41773", "CVE-2021-42013"],  # traversal → incomplete-fix chain
            ["CVE-2023-25690", "CVE-2024-38476"],   # smuggling → SSRF via proxy
        ],
    },
}


# ─────────────────────────────────────────────────────────────────────────────
# Helpers for downstream consumers
# ─────────────────────────────────────────────────────────────────────────────

def get_all_cves_in_profiles():
    """Returns set of all CVE IDs mentioned across all profiles."""
    cves = set()
    for profile in STACK_PROFILES.values():
        for item in profile.get("high_confidence", []):
            cves.add(item["cve"])
        for items in profile.get("conditional", {}).values():
            for item in items:
                cves.add(item["cve"])
        for rule in profile.get("negative_rules", []):
            cves.update(rule.get("absent_cves", []))
            cves.update(rule.get("still_assess", []))
        for group in profile.get("remediation_ties", []):
            cves.update(group.get("cves", []))
        for chain in profile.get("attack_chains", []):
            cves.update(chain)
        cves.update(profile.get("independent_assess", []))
    return cves


def get_profile_for_cve(cve_id):
    """Returns list of profile keys that mention a given CVE."""
    results = []
    for key, profile in STACK_PROFILES.items():
        all_cves = set()
        for item in profile.get("high_confidence", []):
            all_cves.add(item["cve"])
        for items in profile.get("conditional", {}).values():
            for item in items:
                all_cves.add(item["cve"])
        if cve_id in all_cves:
            results.append(key)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# Auto-generate candidate profiles from KEV clusters + NVD product data
# ─────────────────────────────────────────────────────────────────────────────
# This mines data/raw_kev_clusters.json and data/raw_nvd.json to discover
# product-CVE groups that don't already have a manual profile.
# Auto-generated profiles carry `_auto_generated: True` so humans can review.

_DATA_DIR = Path(__file__).parent / "data"

# Minimum CVEs needed to form a candidate profile
_MIN_PROFILE_CVES = 3
# Maximum CVEs from a single product to avoid noise
_MAX_PROFILE_CVES = 25


def _load_json_safe(path: Path):
    if not path.exists():
        return []
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return []


def _existing_profile_cves() -> set[str]:
    """Collect all CVEs already covered by manual profiles."""
    cves = set()
    for profile in STACK_PROFILES.values():
        for item in profile.get("high_confidence", []):
            cves.add(item["cve"])
        for items in profile.get("conditional", {}).values():
            for item in items:
                cves.add(item["cve"])
        for chain in profile.get("attack_chains", []):
            cves.update(chain)
    return cves


def _existing_profile_indicators() -> set[str]:
    """Collect all indicator keywords already used."""
    indicators = set()
    for profile in STACK_PROFILES.values():
        for ind in profile.get("indicators", []):
            indicators.add(ind.lower())
    return indicators


def _product_key_to_display(product_key: str) -> str:
    """Convert 'vendor:product' → 'Vendor Product' display name."""
    parts = product_key.replace("_", " ").replace(":", " ").split()
    return " ".join(p.capitalize() for p in parts)


def auto_generate_stack_profiles() -> dict[str, dict]:
    """
    Mine KEV clusters + NVD data to auto-generate candidate stack profiles
    for products not already covered by manual profiles.

    Returns dict of {profile_key: profile_dict} ready for merging.
    """
    existing_cves = _existing_profile_cves()
    existing_indicators = _existing_profile_indicators()
    candidates: dict[str, dict] = {}

    # ── Source 1: NVD product → CVE index ──────────────────────────────────
    nvd_path = _DATA_DIR / "raw_nvd.json"
    nvd_records = _load_json_safe(nvd_path)
    if not nvd_records:
        # Try JSONL format
        jsonl_path = _DATA_DIR / "vuln_dataset.jsonl"
        if jsonl_path.exists():
            nvd_records = []
            try:
                with open(jsonl_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            nvd_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
            except OSError:
                pass

    # Build product → CVE map from NVD
    product_cves: dict[str, list[dict]] = defaultdict(list)
    for rec in nvd_records:
        cve_id = rec.get("cve_id", "") or rec.get("id", "")
        if not cve_id:
            continue
        desc = rec.get("description", "")[:200]
        cvss = rec.get("cvss_score", 0)

        # Extract products from affected_software
        software = rec.get("affected_software", [])
        for sw in software:
            if isinstance(sw, str) and len(sw) > 2:
                key = re.sub(r"\s+\d[\d.]*\s*$", "", sw.lower().strip())[:60]
                if key:
                    product_cves[key].append({
                        "cve": cve_id, "desc": desc,
                        "cvss": float(cvss) if cvss else 0.0,
                    })

        # Also check CPE fields
        for field in ("cpe", "cpes", "affected_products"):
            val = rec.get(field)
            if isinstance(val, list):
                for item in val:
                    cpe_str = item if isinstance(item, str) else (
                        item.get("cpe23Uri", item.get("cpe", "")) if isinstance(item, dict) else ""
                    )
                    if cpe_str and ":" in cpe_str:
                        parts = cpe_str.split(":")
                        if len(parts) >= 5:
                            key = f"{parts[3]}:{parts[4]}"
                            product_cves[key].append({
                                "cve": cve_id, "desc": desc,
                                "cvss": float(cvss) if cvss else 0.0,
                            })

    # ── Source 2: KEV cluster vendor groups ────────────────────────────────
    kev_path = _DATA_DIR / "raw_kev_clusters.json"
    kev_data = _load_json_safe(kev_path)
    if isinstance(kev_data, dict):
        for cluster in kev_data.get("vendor_clusters", []):
            stack = cluster.get("stack", "")
            cves = cluster.get("cves", [])
            if stack and len(cves) >= _MIN_PROFILE_CVES:
                key = f"kev_{stack}"
                for cve_id in cves:
                    product_cves[key].append({
                        "cve": cve_id, "desc": f"KEV cluster: {stack}",
                        "cvss": 0.0,
                    })

    # ── Build candidate profiles ───────────────────────────────────────────
    for product_key, cve_entries in product_cves.items():
        # Deduplicate CVEs
        seen = set()
        unique_entries = []
        for entry in cve_entries:
            if entry["cve"] not in seen:
                seen.add(entry["cve"])
                unique_entries.append(entry)

        # Filter: must meet minimum, not too many (noise), some not already covered
        if len(unique_entries) < _MIN_PROFILE_CVES:
            continue

        new_cves = [e for e in unique_entries if e["cve"] not in existing_cves]
        if len(new_cves) < 2:
            continue  # Most already covered by manual profiles

        # Check indicator overlap with existing profiles
        indicator = product_key.replace(":", " ").replace("_", " ").lower()
        indicator_words = set(indicator.split())
        if any(ind in indicator for ind in existing_indicators):
            continue  # Likely already covered

        # Sort by CVSS descending, take top entries
        unique_entries.sort(key=lambda e: e["cvss"], reverse=True)
        top_entries = unique_entries[:_MAX_PROFILE_CVES]

        display_name = _product_key_to_display(product_key)
        profile_key = f"auto_{product_key.replace(':', '_').replace(' ', '_').replace('-', '_')[:40]}"

        # Build high_confidence from highest-CVSS CVEs
        high_conf = [
            {"cve": e["cve"], "reason": e["desc"][:100] or f"Affects {display_name}"}
            for e in top_entries[:min(5, len(top_entries))]
        ]

        # Build simple attack chains from temporally adjacent KEV CVEs
        attack_chains = []
        kev_cves_in_product = []
        if isinstance(kev_data, dict):
            all_kev_cves = set()
            for cluster in kev_data.get("temporal_clusters", []):
                all_kev_cves.update(cluster.get("cves", []))
            kev_cves_in_product = [e["cve"] for e in top_entries if e["cve"] in all_kev_cves]
            if len(kev_cves_in_product) >= 2:
                attack_chains.append(kev_cves_in_product[:3])

        candidates[profile_key] = {
            "display_name":   f"{display_name} (auto-generated)",
            "indicators":     list(indicator_words)[:5],
            "high_confidence": high_conf,
            "conditional":    {},
            "negative_rules": [],
            "remediation_ties": [],
            "attack_chains":  attack_chains,
            "_auto_generated": True,
            "_source_product": product_key,
            "_cve_count":     len(unique_entries),
        }

    if candidates:
        log.info(f"  Auto-generated {len(candidates)} candidate stack profiles")

    return candidates


# ── Auto-append generated profiles at import time ─────────────────────────────
try:
    _auto_profiles = auto_generate_stack_profiles()
    for _k, _v in _auto_profiles.items():
        if _k not in STACK_PROFILES:
            STACK_PROFILES[_k] = _v
except Exception as _e:
    # Non-fatal: data files may not exist yet during first pipeline run
    log.debug(f"Auto-profile generation skipped: {_e}")


if __name__ == "__main__":
    all_cves = get_all_cves_in_profiles()
    print(f"Stack profiles defined: {len(STACK_PROFILES)}")
    print(f"Total CVEs referenced:  {len(all_cves)}")
    for k, v in STACK_PROFILES.items():
        chains = v.get("attack_chains", [])
        print(f"  {k:35s}  high_conf={len(v.get('high_confidence',[]))}  chains={len(chains)}")