/**
 * vuln-graph-backend/server.js
 * ----------------------------
 * Express + Neo4j backend for the vulnerability knowledge graph.
 *
 * FIXES vs previous version:
 *   1. Hardcoded credentials replaced with env var lookup (NEO4J_URI,
 *      NEO4J_USER, NEO4J_PASSWORD). Password is required from env.
 *
 *   2. [NEW] GET /api/cve/:cveId/correlations
 *      Returns direct CORRELATED_WITH / CO_OCCURS_WITH neighbours for a CVE,
 *      ordered by confidence. Used by tool_likely_on_system API path.
 *
 *   3. [NEW] GET /api/cve/:cveId/full
 *      Returns a CVE node plus all its edges in one call: CWE, OWASP,
 *      correlated CVEs, co-occurring CVEs, software affected, and cluster.
 *      Useful for the agent's "what do I know about CVE-X" step.
 *
 *   4. [NEW] GET /api/cwe/:cweId/vulns
 *      CWE-first entry point. Returns all CVEs with that CWE, ordered by
 *      EPSS, plus sibling CWEs in the same cluster. Powers tool_lookup_by_cwe.
 *
 *   5. [NEW] GET /api/cve/:cveId/chain
 *      Returns exploit chain context: CVEs that appear in the same exploit
 *      code or PoC repo (EXPLOIT_CHAIN edges), ordered by confidence.
 *
 *   6. [NEW] GET /api/search?q=...
 *      Basic text search across vuln_id, description fragment. Lets the
 *      frontend search box actually query the graph.
 *
 *   7. GET /api/graph now accepts ?limit= query param (default 300, max 1000).
 *
 *   8. Added /api/health endpoint so tools can probe DB availability.
 */

require('dotenv').config();
const express = require('express');
const neo4j   = require('neo4j-driver');
const cors    = require('cors');

// ── Configuration (FIX 1: env vars, not hardcoded) ─────────────────────────
const NEO4J_URI      = process.env.NEO4J_URI      || 'bolt://localhost:7687';
const NEO4J_USER     = process.env.NEO4J_USER     || 'neo4j';
const NEO4J_PASSWORD = process.env.NEO4J_PASSWORD || '';
const PORT           = parseInt(process.env.PORT  || '3000', 10);
const APP_ENV        = (process.env.APP_ENV || 'development').toLowerCase();
const API_KEY        = (process.env.API_KEY || '').trim();
const RATE_LIMIT_WINDOW_MS = Math.max(parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10), 1000);
const RATE_LIMIT_MAX = Math.max(parseInt(process.env.RATE_LIMIT_MAX || '120', 10), 1);
const CORS_ALLOWLIST = (process.env.CORS_ALLOWLIST || 'http://localhost:3000,http://127.0.0.1:3000')
    .split(',')
    .map(x => x.trim())
    .filter(Boolean);

if (!NEO4J_PASSWORD) {
    throw new Error('NEO4J_PASSWORD is required.');
}

const app    = express();
const driver = neo4j.driver(NEO4J_URI, neo4j.auth.basic(NEO4J_USER, NEO4J_PASSWORD));

app.use(cors({
    origin(origin, cb) {
        if (!origin || CORS_ALLOWLIST.includes(origin)) return cb(null, true);
        return cb(new Error('CORS origin not allowed'));
    },
    methods: ['GET', 'POST', 'OPTIONS'],
    allowedHeaders: ['Content-Type', 'X-API-Key'],
}));
app.use(express.json());


// ── Helpers ────────────────────────────────────────────────────────────────

function parseProperties(properties) {
    if (!properties) return {};
    const parsed = {};
    for (const [key, value] of Object.entries(properties)) {
        parsed[key] = neo4j.isInt(value) ? value.toNumber() : value;
    }
    return parsed;
}

function getDisplayId(properties, label) {
    return (
        properties.vuln_id     ||
        properties.cwe_id      ||
        properties.owasp_id    ||
        properties.software_key||
        properties.cluster_id  ||
        properties.profile_id  ||
        properties.indicator_key||
        'Unknown ID'
    );
}

function neo4jError(res, error) {
    console.error('Neo4j Error:', error.message);
    res.status(500).json({ error: 'internal_error', message: 'Internal server error' });
}

async function runQuery(session, cypher, params = {}) {
    const result = await session.run(cypher, params);
    return result.records;
}

const CVE_RE = /^CVE-\d{4}-\d+$/i;
const CWE_RE = /^CWE-\d+$/i;

function normalizeCve(cveId) {
    const cve = String(cveId || '').toUpperCase().trim();
    return CVE_RE.test(cve) ? cve : null;
}

function normalizeCwe(cweId) {
    let cwe = String(cweId || '').toUpperCase().trim();
    if (!cwe.startsWith('CWE-')) cwe = `CWE-${cwe}`;
    return CWE_RE.test(cwe) ? cwe : null;
}

function parseBoundedInt(value, fallback, min, max) {
    const parsed = parseInt(value, 10);
    if (Number.isNaN(parsed)) return fallback;
    return Math.min(Math.max(parsed, min), max);
}

const rateBuckets = new Map();
function rateLimit(req, res, next) {
    const now = Date.now();
    const key = req.ip || req.connection?.remoteAddress || 'unknown';
    const bucket = rateBuckets.get(key) || { count: 0, resetAt: now + RATE_LIMIT_WINDOW_MS };
    if (now > bucket.resetAt) {
        bucket.count = 0;
        bucket.resetAt = now + RATE_LIMIT_WINDOW_MS;
    }
    bucket.count += 1;
    rateBuckets.set(key, bucket);
    if (bucket.count > RATE_LIMIT_MAX) {
        return res.status(429).json({ error: 'rate_limited', message: 'Too many requests' });
    }
    return next();
}

function requireApiKey(req, res, next) {
    if (!API_KEY) return next();
    const clientKey = (req.header('x-api-key') || '').trim();
    if (clientKey !== API_KEY) {
        return res.status(401).json({ error: 'unauthorized', message: 'Invalid API key' });
    }
    return next();
}


// ── GET /api/health ────────────────────────────────────────────────────────
// FIX 8: Quick probe so callers can check DB availability without a full query
app.get('/api/health', async (_req, res) => {
    const session = driver.session();
    try {
        await session.run('RETURN 1');
        res.json({ status: 'ok', neo4j: NEO4J_URI });
    } catch (e) {
        res.status(503).json({ status: 'unavailable', error: e.message });
    } finally {
        await session.close();
    }
});

// Protect all API routes except health (registered above).
app.use('/api', rateLimit, requireApiKey);


// ── GET /api/graph?limit=300 ───────────────────────────────────────────────
// FIX 7: Accepts optional limit param
app.get('/api/graph', async (req, res) => {
    const session = driver.session();
    const limit   = parseBoundedInt(req.query.limit || '300', 300, 1, 1000);
    try {
        const records = await runQuery(session, `
            MATCH (n)
            WITH n LIMIT $limit
            OPTIONAL MATCH (n)-[r]->(m)
            RETURN n, r, m
        `, { limit: neo4j.int(limit) });

        const nodesMap = new Map();
        const links    = [];

        records.forEach(record => {
            const n = record.get('n');
            const r = record.get('r');
            const m = record.get('m');

            if (n && !nodesMap.has(n.elementId)) {
                const props = parseProperties(n.properties);
                nodesMap.set(n.elementId, {
                    id: n.elementId,
                    displayId: getDisplayId(props, n.labels[0]),
                    label: n.labels[0] || 'Unknown',
                    properties: props,
                });
            }
            if (r && m) {
                if (!nodesMap.has(m.elementId)) {
                    const props = parseProperties(m.properties);
                    nodesMap.set(m.elementId, {
                        id: m.elementId,
                        displayId: getDisplayId(props, m.labels[0]),
                        label: m.labels[0] || 'Unknown',
                        properties: props,
                    });
                }
                links.push({
                    source:     n.elementId,
                    target:     m.elementId,
                    type:       r.type,
                    properties: parseProperties(r.properties),
                });
            }
        });

        res.json({ nodes: Array.from(nodesMap.values()), links });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── GET /api/cve/:cveId/correlations?top_k=15 ─────────────────────────────
// FIX 2: NEW — returns direct graph neighbours for a CVE
app.get('/api/cve/:cveId/correlations', async (req, res) => {
    const session = driver.session();
    const cveId   = normalizeCve(req.params.cveId);
    const topK    = parseBoundedInt(req.query.top_k || '15', 15, 1, 50);
    if (!cveId) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cve', message: 'Expected CVE-YYYY-NNNN format' });
    }
    try {
        const records = await runQuery(session, `
            MATCH (v:Vulnerability {vuln_id: $cveId})
                  -[r:CORRELATED_WITH|CO_OCCURS_WITH]-(related:Vulnerability)
            RETURN related.vuln_id                                    AS cve_id,
                   type(r)                                            AS rel_type,
                   coalesce(r.max_score, r.max_confidence, 0.0)       AS confidence,
                   coalesce(r.signals, [])                            AS signals,
                   coalesce(r.reasons, [])                            AS reasons,
                   coalesce(related.cvss_score, 0.0)                  AS cvss,
                   coalesce(related.epss_score, 0.0)                  AS epss,
                   coalesce(related.confirmed_exploited, false)        AS kev
            ORDER BY confidence DESC
            LIMIT $topK
        `, { cveId, topK: neo4j.int(topK) });

        const results = records.map(r => ({
            cve_id:     r.get('cve_id'),
            rel_type:   r.get('rel_type'),
            confidence: r.get('confidence'),
            signals:    r.get('signals'),
            reasons:    r.get('reasons'),
            cvss:       r.get('cvss'),
            epss:       r.get('epss'),
            kev:        r.get('kev'),
        }));

        res.json({ query_cve: cveId, count: results.length, results });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── GET /api/cve/:cveId/full ───────────────────────────────────────────────
// FIX 3: NEW — returns a CVE's complete neighbourhood in one call
app.get('/api/cve/:cveId/full', async (req, res) => {
    const session = driver.session();
    const cveId   = normalizeCve(req.params.cveId);
    if (!cveId) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cve', message: 'Expected CVE-YYYY-NNNN format' });
    }
    try {
        // Core node
        const nodeRec = await runQuery(session, `
            MATCH (v:Vulnerability {vuln_id: $cveId})
            OPTIONAL MATCH (v)-[:HAS_CWE]->(w:CWE)
            OPTIONAL MATCH (v)-[:MAPS_TO_OWASP]->(o:OWASPCategory)
            RETURN v,
                   collect(DISTINCT w.cwe_id) AS cwes,
                   collect(DISTINCT o.owasp_id) AS owasps
        `, { cveId });

        if (!nodeRec.length) {
            return res.status(404).json({ error: `CVE ${cveId} not found in graph` });
        }

        const vuln  = parseProperties(nodeRec[0].get('v').properties);
        const cwes  = nodeRec[0].get('cwes');
        const owasps= nodeRec[0].get('owasps');

        // Correlations
        const corrRec = await runQuery(session, `
            MATCH (v:Vulnerability {vuln_id: $cveId})
                  -[r:CORRELATED_WITH|CO_OCCURS_WITH]-(related:Vulnerability)
            RETURN related.vuln_id AS cve_id,
                   type(r) AS rel_type,
                   coalesce(r.max_score, r.max_confidence, 0.0) AS confidence
            ORDER BY confidence DESC LIMIT 20
        `, { cveId });

        // Software
        const swRec = await runQuery(session, `
            MATCH (v:Vulnerability {vuln_id: $cveId})-[:AFFECTS_SOFTWARE]->(s:Software)
            RETURN s.software_key AS software
            LIMIT 15
        `, { cveId });

        res.json({
            cve_id:       cveId,
            properties:   vuln,
            cwes,
            owasp:        owasps,
            software:     swRec.map(r => r.get('software')),
            correlations: corrRec.map(r => ({
                cve_id:     r.get('cve_id'),
                rel_type:   r.get('rel_type'),
                confidence: r.get('confidence'),
            })),
        });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── GET /api/cwe/:cweId/vulns?top_k=20 ────────────────────────────────────
// FIX 4: NEW — CWE-first lookup, powers tool_lookup_by_cwe
app.get('/api/cwe/:cweId/vulns', async (req, res) => {
    const session = driver.session();
    const cweId   = normalizeCwe(req.params.cweId);
    const topK    = parseBoundedInt(req.query.top_k || '20', 20, 1, 100);
    if (!cweId) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cwe', message: 'Expected CWE-NNN format' });
    }

    try {
        // CVEs with this CWE
        const vulnRec = await runQuery(session, `
            MATCH (w:CWE {cwe_id: $cweId})<-[:HAS_CWE]-(v:Vulnerability)
            OPTIONAL MATCH (v)-[:MAPS_TO_OWASP]->(o:OWASPCategory)
            RETURN v.vuln_id                             AS cve_id,
                   coalesce(v.cvss_score, 0.0)            AS cvss,
                   coalesce(v.epss_score, 0.0)            AS epss,
                   coalesce(v.confirmed_exploited, false)  AS kev,
                   o.owasp_id                             AS owasp,
                   v.description                          AS description
            ORDER BY epss DESC, cvss DESC
            LIMIT $topK
        `, { cweId, topK: neo4j.int(topK) });

        // Sibling CWEs in the same cluster
        const clusterRec = await runQuery(session, `
            MATCH (w:CWE {cwe_id: $cweId})<-[:CONTAINS_CWE]-(cluster:CWECluster)
                  -[:CONTAINS_CWE]->(sibling:CWE)
            WHERE sibling.cwe_id <> $cweId
            RETURN sibling.cwe_id    AS sibling_cwe,
                   cluster.cluster_id AS cluster_id
            LIMIT 10
        `, { cweId });

        res.json({
            cwe_id:       cweId,
            vuln_count:   vulnRec.length,
            vulns:        vulnRec.map(r => ({
                cve_id:      r.get('cve_id'),
                cvss:        r.get('cvss'),
                epss:        r.get('epss'),
                kev:         r.get('kev'),
                owasp:       r.get('owasp'),
                description: (r.get('description') || '').slice(0, 200),
            })),
            cluster:      clusterRec.length
                ? clusterRec[0].get('cluster_id')
                : null,
            sibling_cwes: clusterRec.map(r => r.get('sibling_cwe')),
        });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── GET /api/cve/:cveId/chain ──────────────────────────────────────────────
// FIX 5: NEW — exploit chain context
app.get('/api/cve/:cveId/chain', async (req, res) => {
    const session = driver.session();
    const cveId   = normalizeCve(req.params.cveId);
    if (!cveId) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cve', message: 'Expected CVE-YYYY-NNNN format' });
    }
    try {
        const records = await runQuery(session, `
            MATCH (v:Vulnerability {vuln_id: $cveId})
                  -[r:EXPLOIT_CHAIN|CHAINED_WITH]->(target:Vulnerability)
            RETURN target.vuln_id                            AS target_cve,
                   r.chain_role                              AS role,
                   coalesce(r.confidence, 0.5)               AS confidence,
                   coalesce(r.source, 'unknown')             AS source
            ORDER BY confidence DESC
            LIMIT 20
        `, { cveId });

        res.json({
            query_cve: cveId,
            chains:    records.map(r => ({
                target_cve: r.get('target_cve'),
                role:       r.get('role'),
                confidence: r.get('confidence'),
                source:     r.get('source'),
            })),
        });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── GET /api/search?q=CVE-2021 ────────────────────────────────────────────
// FIX 6: NEW — basic text search across vulnerability IDs and descriptions
app.get('/api/search', async (req, res) => {
    const session = driver.session();
    const q       = (req.query.q || '').trim().toUpperCase();
    if (!q || q.length < 3) {
        return res.status(400).json({ error: 'Query must be at least 3 characters' });
    }
    try {
        const records = await runQuery(session, `
            MATCH (v:Vulnerability)
            WHERE v.vuln_id CONTAINS $q
               OR toUpper(coalesce(v.description, '')) CONTAINS $q
               OR toUpper(coalesce(v.vulnerability_name, '')) CONTAINS $q
            RETURN v.vuln_id                            AS cve_id,
                   v.vulnerability_name                 AS name,
                   coalesce(v.cvss_score, 0.0)           AS cvss,
                   coalesce(v.epss_score, 0.0)           AS epss,
                   coalesce(v.confirmed_exploited, false) AS kev,
                   left(coalesce(v.description, ''), 200) AS description
            ORDER BY epss DESC
            LIMIT 25
        `, { q });

        res.json({
            query: q,
            count: records.length,
            results: records.map(r => ({
                cve_id:      r.get('cve_id'),
                name:        r.get('name'),
                cvss:        r.get('cvss'),
                epss:        r.get('epss'),
                kev:         r.get('kev'),
                description: r.get('description'),
            })),
        });
    } catch (e) {
        neo4jError(res, e);
    } finally {
        await session.close();
    }
});

// ── POST /api/graphrag/query ────────────────────────────────────────────────
// Strict JSON contract response for agent-to-agent retrieval.
app.post('/api/graphrag/query', async (req, res) => {
    const session = driver.session();
    const body = req.body || {};
    const query = String(body.query || '').trim();
    const topK = parseBoundedInt(body.top_k || 12, 12, 1, 25);
    const entity = body.entity || {};

    if (!query) {
        await session.close();
        return res.status(400).json({ error: 'invalid_query', message: 'query is required' });
    }

    let entityType = String(entity.type || '').toLowerCase().trim();
    let entityId = String(entity.id || '').toUpperCase().trim();
    if (!entityType || !entityId) {
        const cveMatch = query.toUpperCase().match(/CVE-\d{4}-\d+/);
        const cweMatch = query.toUpperCase().match(/CWE-\d+/);
        if (cveMatch) {
            entityType = 'cve';
            entityId = cveMatch[0];
        } else if (cweMatch) {
            entityType = 'cwe';
            entityId = cweMatch[0];
        } else {
            entityType = 'unknown';
            entityId = '';
        }
    }

    if (entityType === 'cve' && !normalizeCve(entityId)) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cve', message: 'Expected CVE-YYYY-NNNN format' });
    }
    if (entityType === 'cwe' && !normalizeCwe(entityId)) {
        await session.close();
        return res.status(400).json({ error: 'invalid_cwe', message: 'Expected CWE-NNN format' });
    }

    try {
        let rows = [];
        if (entityType === 'cve' && entityId) {
            rows = await runQuery(session, `
                MATCH (v:Vulnerability {vuln_id: $entityId})
                      -[r:CORRELATED_WITH|CO_OCCURS_WITH]-(related:Vulnerability)
                RETURN related.vuln_id AS cve_id,
                       coalesce(r.max_score, r.max_confidence, 0.0) AS score,
                       type(r) AS rel_type,
                       coalesce(r.signals, []) AS signals,
                       coalesce(r.reasons, []) AS reasons
                ORDER BY score DESC
                LIMIT $topK
            `, { entityId, topK: neo4j.int(topK) });
        } else if (entityType === 'cwe' && entityId) {
            rows = await runQuery(session, `
                MATCH (w:CWE {cwe_id: $entityId})<-[:HAS_CWE]-(v:Vulnerability)
                OPTIONAL MATCH (v)-[r:CORRELATED_WITH|CO_OCCURS_WITH]-(peer:Vulnerability)
                RETURN coalesce(peer.vuln_id, v.vuln_id) AS cve_id,
                       coalesce(r.max_score, r.max_confidence, v.epss_score, 0.2) AS score,
                       coalesce(type(r), 'HAS_CWE') AS rel_type,
                       CASE WHEN r IS NULL THEN ['shared_cwe:' + $entityId] ELSE coalesce(r.signals, []) END AS signals,
                       coalesce(r.reasons, []) AS reasons
                ORDER BY score DESC
                LIMIT $topK
            `, { entityId, topK: neo4j.int(topK) });
        } else {
            rows = await runQuery(session, `
                MATCH (v:Vulnerability)
                WHERE toUpper(coalesce(v.description, '')) CONTAINS $q
                   OR toUpper(coalesce(v.vulnerability_name, '')) CONTAINS $q
                RETURN v.vuln_id AS cve_id,
                       coalesce(v.epss_score, 0.2) AS score,
                       'TEXT_MATCH' AS rel_type,
                       [] AS signals,
                       [] AS reasons
                ORDER BY score DESC
                LIMIT $topK
            `, { q: query.toUpperCase(), topK: neo4j.int(topK) });
        }

        const evidence = rows.map((r, idx) => {
            const score = Math.max(0, Math.min(1, Number(r.get('score') || 0)));
            const relType = r.get('rel_type') || '';
            const tier = relType === 'CORRELATED_WITH' || relType === 'CO_OCCURS_WITH' || relType === 'HAS_CWE'
                ? 'direct'
                : 'inferred';
            return {
                rank: idx + 1,
                cve_id: r.get('cve_id'),
                likelihood: Number(score.toFixed(3)),
                evidence_tier: tier,
                rel_type: relType,
                signals: r.get('signals') || [],
                reasons: r.get('reasons') || [],
                inferred_from: tier === 'direct' ? [] : ['text_similarity'],
            };
        });

        const directEvidence = evidence.filter(e => e.evidence_tier === 'direct').slice(0, topK);
        const inferredCandidates = evidence.filter(e => e.evidence_tier !== 'direct').slice(0, topK);
        const top = evidence.slice(0, 5);
        const overall = top.length
            ? Number((top.reduce((acc, row) => acc + row.likelihood, 0) / top.length).toFixed(3))
            : 0.0;

        const hitlReasons = [];
        if (inferredCandidates.length > directEvidence.length) {
            hitlReasons.push('Inferred evidence dominates direct evidence.');
        }
        if (overall < 0.45) {
            hitlReasons.push('Overall confidence below threshold.');
        }
        if (entityType === 'cve' && directEvidence.length < 2) {
            hitlReasons.push('High-risk CVE context has weak supporting evidence.');
        }

        const citations = evidence.slice(0, topK).map((row, idx) => ({
            citation_id: `cit-${idx + 1}-${String(row.cve_id || 'unknown').replace(/[^A-Z0-9-]/ig, '')}`,
            source_type: row.rel_type === 'TEXT_MATCH' ? 'text' : 'graph',
            entity_id: row.cve_id,
            snippet: (row.signals && row.signals.length)
                ? row.signals.slice(0, 2).join(', ')
                : `${row.rel_type} evidence`,
            metadata: { tier: row.evidence_tier },
        }));

        return res.json({
            status: hitlReasons.length ? 'needs_human_review' : 'ok',
            query,
            entity: { type: entityType || 'unknown', id: entityId || '' },
            direct_evidence: directEvidence,
            inferred_candidates: inferredCandidates,
            citations: citations,
            confidence_summary: {
                overall,
                rationale: `Graph retrieval returned ${evidence.length} candidates.`,
            },
            hitl: { required: hitlReasons.length > 0, reasons: hitlReasons },
            recommended_actions: hitlReasons.length
                ? []
                : [
                    'Validate direct evidence against environment inventory.',
                    'Test inferred candidates before remediation prioritization.',
                ],
        });
    } catch (e) {
        return neo4jError(res, e);
    } finally {
        await session.close();
    }
});


// ── Start ──────────────────────────────────────────────────────────────────
app.listen(PORT, () => {
    console.log(`\n✅ Backend running at http://localhost:${PORT}`);
    console.log(`   Neo4j: ${NEO4J_URI}`);
    console.log(`   API auth: ${API_KEY ? 'enabled' : 'disabled'}`);
    console.log('\nEndpoints:');
    console.log(`  GET /api/health`);
    console.log(`  GET /api/graph?limit=300`);
    console.log(`  GET /api/cve/:cveId/correlations?top_k=15`);
    console.log(`  GET /api/cve/:cveId/full`);
    console.log(`  GET /api/cve/:cveId/chain`);
    console.log(`  GET /api/cwe/:cweId/vulns?top_k=20`);
    console.log(`  GET /api/search?q=CVE-2021`);
    console.log(`  POST /api/graphrag/query`);
});
