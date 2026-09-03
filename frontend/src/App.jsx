import { useEffect, useState, useMemo } from "react";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from "recharts";
import {
  ShieldCheck,
  ArrowRight,
  Loader2,
  Search,
  AlertTriangle,
  Sparkles,
} from "lucide-react";
import { getIncidents, getIncidentStats, getIncident, investigateIncident } from "./api";

const TRUST_COLOR = {
  HIGH_TRUST: "#3FB88A",
  MEDIUM_TRUST: "#E8A23A",
  LOW_TRUST: "#E5484D",
};

const ACTION_COLOR = {
  PROCEED: "#3FB88A",
  REVIEW: "#E8A23A",
  FALLBACK: "#E5484D",
};

const COMPONENT_LABELS = {
  data_quality: "Data quality",
  drift: "Drift",
  signal_consistency: "Signal conflict",
  model_confidence: "Model confidence",
  decision_robustness: "Decision robustness",
};

// Component scores are colored by their own value, not a blanket trust color -
// so a strong sub-score and a weak one are visually distinct at a glance.
function scoreColor(value) {
  if (value >= 70) return "#3FB88A";
  if (value >= 45) return "#E8A23A";
  return "#E5484D";
}

// Picks the weakest real component score so the hero narrative line can name
// an actual cause, computed from this incident's own data - never invented.
function weakestComponent(scores) {
  const entries = Object.entries(scores).filter(([, v]) => v != null);
  if (!entries.length) return null;
  const [key, value] = entries.reduce((min, e) => (e[1] < min[1] ? e : min));
  return { label: COMPONENT_LABELS[key] || key, value };
}

function TrustBadge({ level, subtle }) {
  const color = TRUST_COLOR[level] || "#8A8F96";
  if (subtle) {
    return (
      <span className="inline-flex items-center gap-1.5 text-xs text-muted">
        <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
        {level.replace("_TRUST", "").toLowerCase()} trust
      </span>
    );
  }
  return (
    <span className="inline-flex items-center gap-1.5 text-xs font-medium" style={{ color }}>
      <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: color }} />
      {level.replace("_", " ")}
    </span>
  );
}

function Stat({ label, value, sub, color }) {
  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-muted mb-1.5">{label}</div>
      <div className="font-mono text-[26px] leading-none" style={{ color: color || "#EDEEF0" }}>
        {value}
      </div>
      {sub && <div className="text-xs text-muted mt-1.5">{sub}</div>}
    </div>
  );
}

function TrustDistribution({ breakdown, total }) {
  const entries = ["HIGH_TRUST", "MEDIUM_TRUST", "LOW_TRUST"].map((level) => ({
    level,
    label: level.replace("_TRUST", "").toLowerCase(),
    count: breakdown[level] || 0,
    color: TRUST_COLOR[level],
  }));

  return (
    <div>
      <div className="text-[11px] uppercase tracking-wide text-muted mb-1.5">Trust distribution</div>
      <div className="flex h-1.5 rounded-full overflow-hidden mb-2 bg-border">
        {entries.map((e) => (
          <div
            key={e.level}
            style={{ width: `${total ? (e.count / total) * 100 : 0}%`, backgroundColor: e.color }}
          />
        ))}
      </div>
      <div className="flex items-center gap-3 flex-wrap text-xs text-muted">
        {entries.map((e) => (
          <span key={e.level} className="flex items-center gap-1.5">
            <span className="w-1.5 h-1.5 rounded-full" style={{ backgroundColor: e.color }} />
            {e.label} {total ? (e.count / total * 100).toFixed(e.count / total * 100 < 1 ? 2 : 0) : 0}%
          </span>
        ))}
      </div>
    </div>
  );
}

function StatsBar({ stats }) {
  if (!stats) return null;

  return (
    <div className="grid grid-cols-2 lg:grid-cols-5 gap-8 pb-8 mb-8 border-b border-border">
      <Stat label="Transactions evaluated" value={stats.transactions_evaluated.toLocaleString()} />
      <Stat
        label="Flagged"
        value={stats.transactions_flagged.toLocaleString()}
        sub={`${(stats.flag_rate * 100).toFixed(1)}% flag rate`}
        color={TRUST_COLOR.LOW_TRUST}
      />
      <Stat label="Avg trust score" value={stats.average_trust_score.toFixed(1)} color={TRUST_COLOR.HIGH_TRUST} />
      <Stat
        label="Need attention"
        value={(stats.action_breakdown.REVIEW + stats.action_breakdown.FALLBACK).toLocaleString()}
        sub={`${stats.action_breakdown.PROCEED.toLocaleString()} proceeded`}
        color={TRUST_COLOR.MEDIUM_TRUST}
      />
      <TrustDistribution breakdown={stats.trust_level_breakdown || {}} total={stats.transactions_evaluated} />
    </div>
  );
}

function IncidentRow({ incident, selected, onClick }) {
  return (
    <button
      onClick={onClick}
      className={`w-full text-left px-3 py-3 border-b border-border transition-colors ${
        selected ? "bg-surface" : "hover:bg-surface/60"
      }`}
    >
      <div className="flex items-center justify-between mb-1">
        <span className="font-mono text-sm text-ink">{incident.transaction_id}</span>
        <span className="font-mono text-sm" style={{ color: TRUST_COLOR[incident.trust_level] }}>
          {incident.trust_score}
        </span>
      </div>
      <div className="flex items-center justify-between">
        <TrustBadge level={incident.trust_level} subtle />
        <span className="text-xs text-muted" style={{ color: ACTION_COLOR[incident.action] }}>
          {incident.action.toLowerCase()}
        </span>
      </div>
    </button>
  );
}

function ComponentScores({ scores }) {
  const data = Object.entries(scores).map(([key, value]) => ({
    name: COMPONENT_LABELS[key] || key,
    value: value ?? 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={170}>
      <BarChart data={data} layout="vertical" margin={{ left: 130, right: 36 }}>
        <XAxis type="number" domain={[0, 100]} hide />
        <YAxis
          type="category"
          dataKey="name"
          width={130}
          tick={{ fill: "#8A8F96", fontSize: 13 }}
          axisLine={false}
          tickLine={false}
        />
        <Tooltip
          cursor={{ fill: "transparent" }}
          contentStyle={{ background: "#121417", border: "0.5px solid #1E2125", fontSize: 12 }}
          labelStyle={{ color: "#EDEEF0" }}
          itemStyle={{ color: "#EDEEF0" }}
        />
        <Bar dataKey="value" radius={[2, 2, 2, 2]} barSize={3} label={{ position: "right", fill: "#8A8F96", fontSize: 13 }}>
          {data.map((entry, i) => (
            <Cell key={i} fill={scoreColor(entry.value)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

function CounterfactualCard({ cf }) {
  if (!cf || cf.original_decision == null) {
    return <div className="text-muted text-sm">No counterfactual data for this transaction.</div>;
  }

  return (
    <div>
      <div className="flex items-center gap-6 flex-wrap">
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted mb-1">Original</div>
          <div className="font-mono text-lg text-ink">{cf.original_decision}</div>
          <div className="font-mono text-xs text-muted mt-0.5">
            {cf.original_fraud_probability?.toFixed(1)}% fraud probability
          </div>
        </div>
        <ArrowRight className="text-muted flex-shrink-0" size={16} />
        <div>
          <div className="text-[11px] uppercase tracking-wide text-muted mb-1">After nudge</div>
          <div className="font-mono text-lg" style={{ color: cf.flipped ? "#E5484D" : "#EDEEF0" }}>
            {cf.counterfactual_decision}
          </div>
          <div className="font-mono text-xs text-muted mt-0.5">
            {cf.counterfactual_fraud_probability?.toFixed(1)}% fraud probability
          </div>
        </div>
      </div>
      {cf.flipped && (
        <div className="mt-4 pt-4 border-t border-border text-xs text-muted">
          Recommendation flips when nudging <span className="font-mono text-ink">{cf.changed_feature}</span>{" "}
          {cf.changed_direction} 1 standard deviation.
        </div>
      )}
    </div>
  );
}

function DecisionVsTrust({ incident, onInvestigate, investigating }) {
  const cf = incident.counterfactual;
  const trustColor = TRUST_COLOR[incident.trust_level];
  const hasOriginal = cf && cf.original_decision != null;
  const weak = weakestComponent(incident.component_scores);

  return (
    <div className="mb-10">
      <div className="text-sm text-muted mb-4">
        {weak
          ? `The model was confident. SignalGuard wasn't — ${weak.label.toLowerCase()} pulled the score down.`
          : "The model was confident. SignalGuard wasn't."}
      </div>
      <div className="flex items-baseline gap-6 flex-wrap mb-6">
        <div>
          <div className="font-mono text-[44px] leading-none tracking-tight text-ink">
            {hasOriginal ? cf.original_decision : "—"}
          </div>
          <div className="text-xs text-muted mt-2">
            {hasOriginal ? `${cf.original_fraud_probability?.toFixed(1)}% fraud probability` : "no model decision on file"}
          </div>
        </div>
        <ArrowRight className="text-[#4B5157]" size={26} />
        <div>
          <div className="font-mono text-[44px] leading-none tracking-tight" style={{ color: trustColor }}>
            {incident.trust_score}
          </div>
          <div className="text-xs text-muted mt-2">
            {incident.trust_level.replace("_TRUST", "").toLowerCase()} trust · {incident.action.toLowerCase()}
          </div>
        </div>
      </div>
      <button
        onClick={onInvestigate}
        disabled={investigating}
        className="inline-flex items-center gap-2 bg-accent text-[#04241A] text-sm font-medium px-4 py-2.5 rounded-lg transition-opacity disabled:opacity-60 hover:opacity-90"
      >
        {investigating ? <Loader2 size={16} className="animate-spin" /> : <Sparkles size={16} />}
        {investigating ? "Investigating" : "Investigate with AI"}
      </button>
    </div>
  );
}

function IncidentDetail({ incident, onInvestigate, investigating, investigation }) {
  if (!incident) {
    return (
      <div className="flex-1 flex items-center justify-center text-muted text-sm">
        Select an incident to see the full trust breakdown.
      </div>
    );
  }

  return (
    <div className="flex-1 overflow-y-auto px-8 py-8">
      <div className="mb-8">
        <div className="font-mono text-muted text-sm mb-1">{incident.transaction_id}</div>
        <TrustBadge level={incident.trust_level} />
      </div>

      <DecisionVsTrust incident={incident} onInvestigate={onInvestigate} investigating={investigating} />

      <div className="mb-10 pt-8 border-t border-border">
        <div className="text-ink text-sm font-medium mb-4">Why trust dropped</div>
        <ComponentScores scores={incident.component_scores} />
      </div>

      <div className="mb-10 pt-8 border-t border-border">
        <div className="text-ink text-sm font-medium mb-4">Counterfactual test</div>
        <CounterfactualCard cf={incident.counterfactual} />
      </div>

      <div className="mb-10 pt-8 border-t border-border">
        <div className="text-ink text-sm font-medium mb-4">
          Evidence <span className="text-muted font-normal">({incident.n_evidence_items})</span>
        </div>
        <div className="space-y-3">
          {incident.evidence.map((item, i) => (
            <div key={i} className="pl-3 border-l-2" style={{ borderColor: "#1E2125" }}>
              <div className="text-[11px] uppercase tracking-wide text-muted mb-1 font-mono">{item.checker}</div>
              <div className="text-sm text-ink">{item.issue}</div>
            </div>
          ))}
        </div>
      </div>

      {investigation && (
        <div className="pt-8 border-t border-border">
          <div className="text-ink text-sm font-medium mb-4 flex items-center gap-2">
            AI investigation
            <span className="font-mono text-[11px] text-muted">{investigation.model}</span>
          </div>
          <div className="text-sm text-ink whitespace-pre-wrap leading-relaxed">
            {investigation.explanation || JSON.stringify(investigation, null, 2)}
          </div>
          {investigation.cached && <div className="mt-3 text-xs text-muted">Served from cache</div>}
        </div>
      )}
    </div>
  );
}

export default function App() {
  const [stats, setStats] = useState(null);
  const [incidents, setIncidents] = useState([]);
  const [selectedId, setSelectedId] = useState(null);
  const [selectedIncident, setSelectedIncident] = useState(null);
  const [search, setSearch] = useState("");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [investigating, setInvestigating] = useState(false);
  const [investigation, setInvestigation] = useState(null);

  // The sidebar only ever loads the first 200 incidents (of potentially
  // thousands) up front, so a plain local filter misses any real incident
  // outside that page. remoteLookup covers that gap: once the local filter
  // comes up empty, we ask the backend directly by transaction ID.
  const [remoteLookup, setRemoteLookup] = useState({ status: "idle", incident: null });

  useEffect(() => {
    Promise.all([getIncidentStats(), getIncidents({ limit: 200 })])
      .then(([statsData, incidentsData]) => {
        setStats(statsData);
        setIncidents(incidentsData.incidents || incidentsData);
      })
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false));
  }, []);

  const filtered = useMemo(() => {
    if (!search.trim()) return incidents;
    const q = search.trim().toLowerCase();
    return incidents.filter((i) => i.transaction_id.toLowerCase().includes(q));
  }, [incidents, search]);

  // Debounced fallback: if the locally loaded page has no match for a
  // non-trivial query, ask the backend directly (GET /api/incidents/{id})
  // rather than reporting "0 incidents" for something that may genuinely
  // exist just outside the first 200 loaded.
  useEffect(() => {
    const q = search.trim();

    if (!q || filtered.length > 0) {
      setRemoteLookup({ status: "idle", incident: null });
      return;
    }

    setRemoteLookup({ status: "loading", incident: null });
    const timer = setTimeout(() => {
      getIncident(q)
        .then((incident) => setRemoteLookup({ status: "found", incident }))
        .catch(() => setRemoteLookup({ status: "not_found", incident: null }));
    }, 350);

    return () => clearTimeout(timer);
  }, [search, filtered.length]);

  // What the sidebar actually renders: the local page filter when it has
  // matches, otherwise the direct backend lookup result (if any).
  const displayedIncidents =
    filtered.length > 0
      ? filtered
      : remoteLookup.status === "found" && remoteLookup.incident
        ? [remoteLookup.incident]
        : [];

  const selectIncident = (id) => {
    setSelectedId(id);
    setInvestigation(null);
    getIncident(id).then(setSelectedIncident).catch((err) => setError(err.message));
  };

  const handleInvestigate = () => {
    if (!selectedId) return;
    setInvestigating(true);
    investigateIncident(selectedId)
      .then(setInvestigation)
      .catch((err) => setError(err.message))
      .finally(() => setInvestigating(false));
  };

  if (loading) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center">
        <Loader2 className="animate-spin text-muted" size={28} />
      </div>
    );
  }

  if (error) {
    return (
      <div className="min-h-screen bg-bg flex items-center justify-center px-4">
        <div className="bg-surface border border-border rounded-lg p-6 max-w-md flex items-start gap-3">
          <AlertTriangle className="text-trust-low flex-shrink-0" size={20} />
          <div>
            <div className="text-ink font-medium mb-1">Couldn't reach the backend</div>
            <div className="text-muted text-sm">{error}. Make sure uvicorn is running on http://127.0.0.1:8000.</div>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-bg text-ink font-sans">
      <header className="px-8 pt-8 pb-6">
        <div className="flex items-center gap-2">
          <ShieldCheck className="text-accent" size={19} />
          <span className="text-[17px] font-medium tracking-tight">SignalGuard</span>
        </div>
        <div className="text-muted text-sm mt-1">AI decision trust layer for payment systems</div>
      </header>

      <main className="px-8 pb-8">
        <StatsBar stats={stats} />

        <div className="flex border border-border rounded-xl overflow-hidden" style={{ height: "calc(100vh - 280px)" }}>
          <div className="w-[320px] border-r border-border flex flex-col">
            <div className="p-4 border-b border-border">
              <div className="relative">
                <Search className="absolute left-2.5 top-2.5 text-muted" size={14} />
                <input
                  value={search}
                  onChange={(e) => setSearch(e.target.value)}
                  placeholder="Search transaction ID"
                  className="w-full bg-bg border border-border rounded-md pl-8 pr-3 py-1.5 text-sm text-ink placeholder:text-muted focus:outline-none focus:border-muted"
                />
              </div>
              <div className="text-muted text-xs mt-2">
                {filtered.length > 0
                  ? `${filtered.length.toLocaleString()} incidents`
                  : remoteLookup.status === "loading"
                    ? "Checking full incident list…"
                    : remoteLookup.status === "found"
                      ? "1 incident (found outside the first 200 loaded)"
                      : remoteLookup.status === "not_found"
                        ? "0 incidents"
                        : `${filtered.length.toLocaleString()} incidents`}
              </div>
            </div>
            <div className="flex-1 overflow-y-auto">
              {displayedIncidents.map((incident) => (
                <IncidentRow
                  key={incident.transaction_id}
                  incident={incident}
                  selected={incident.transaction_id === selectedId}
                  onClick={() => selectIncident(incident.transaction_id)}
                />
              ))}
            </div>
          </div>

          <div className="flex-1 bg-surface/40 flex">
            <IncidentDetail
              incident={selectedIncident}
              onInvestigate={handleInvestigate}
              investigating={investigating}
              investigation={investigation}
            />
          </div>
        </div>
      </main>
    </div>
  );
}
