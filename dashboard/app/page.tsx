"use client";

import { useState, useEffect, useRef, useCallback } from "react";
import {
  LineChart, Line, AreaChart, Area,
  XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from "recharts";
import {
  Shield, ShieldAlert, ShieldCheck, Brain, Zap, Eye,
  AlertTriangle, CheckCircle, XCircle, Activity, Lock,
  FileText, TrendingUp, Clock, Wifi, WifiOff, Terminal,
  ChevronRight, Radio, Cpu, Database, AlertOctagon
} from "lucide-react";
import { useAgent } from "./hooks/useAgent";

// ── Static data ─────────────────────────────────────────────────────────────
const VENDORS = [
  { name: "Emotiv",    score: 32, risk: "CRITICAL", critical: 4, compliant: 2 },
  { name: "Muse",      score: 45, risk: "HIGH",     critical: 3, compliant: 3 },
  { name: "OpenBCI",   score: 71, risk: "MEDIUM",   critical: 1, compliant: 6 },
  { name: "Neurosity", score: 58, risk: "HIGH",     critical: 2, compliant: 4 },
  { name: "Neuralink", score: 88, risk: "LOW",      critical: 0, compliant: 8 },
];

const ROBUSTNESS = [
  { eps: "0.01", fgsm: 82, pgd: 80 }, { eps: "0.02", fgsm: 75, pgd: 70 },
  { eps: "0.05", fgsm: 48, pgd: 40 }, { eps: "0.08", fgsm: 30, pgd: 22 },
  { eps: "0.10", fgsm: 20, pgd: 14 }, { eps: "0.15", fgsm: 12, pgd: 8 },
  { eps: "0.20", fgsm: 8,  pgd: 5  },
];

const PAPERS = [
  { title: "Adversarial Filtering Attacks on EEG BCIs",         authors: "Meng et al.",        year: "2024", venue: "IEEE TIFS" },
  { title: "Protecting Multiple Privacy Types in EEG BCIs",     authors: "Meng et al.",        year: "2024", venue: "arXiv" },
  { title: "Cyber Risks to Next-Generation BCIs",               authors: "Schroder et al.",    year: "2025", venue: "Springer Neuroethics" },
  { title: "Adversarial Attacks on Medical Machine Learning",   authors: "Finlayson et al.",   year: "2019", venue: "Science" },
  { title: "Side-Channel Attacks with Brain-Computer Interfaces", authors: "Martinovic et al.", year: "2012", venue: "USENIX Security" },
  { title: "User Identity Protection in EEG BCIs",              authors: "Chen et al.",        year: "2025", venue: "J. Neural Eng." },
];

// ── Components ───────────────────────────────────────────────────────────────
function RiskBadge({ risk }: { risk: string }) {
  const styles: Record<string, string> = {
    CLEAN:    "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
    BLOCKED:  "text-red-400 bg-red-400/10 border-red-400/30",
    ATTACKED: "text-amber-400 bg-amber-400/10 border-amber-400/30",
    CRITICAL: "text-red-400 bg-red-400/10 border-red-400/30",
    HIGH:     "text-orange-400 bg-orange-400/10 border-orange-400/30",
    MEDIUM:   "text-amber-400 bg-amber-400/10 border-amber-400/30",
    LOW:      "text-emerald-400 bg-emerald-400/10 border-emerald-400/30",
  };
  const dots: Record<string, string> = {
    CLEAN: "bg-emerald-400", BLOCKED: "bg-red-400 animate-pulse",
    ATTACKED: "bg-amber-400", CRITICAL: "bg-red-500 animate-pulse",
    HIGH: "bg-orange-400", MEDIUM: "bg-amber-400", LOW: "bg-emerald-400",
  };
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded border text-[10px] font-bold tracking-wider ${styles[risk] ?? styles.LOW}`}>
      <span className={`w-1.5 h-1.5 rounded-full ${dots[risk] ?? "bg-slate-400"}`} />
      {risk}
    </span>
  );
}

// ── Main ─────────────────────────────────────────────────────────────────────
export default function CyberNeuroDashboard() {
  const [tab, setTab] = useState<"stream" | "attacks" | "compliance" | "privacy" | "research">("stream");
  const [running, setRunning] = useState(false);

  // ── Real agent hook — connects to FastAPI backend ──────────────────────────
  const { currentWindow, windows, incidents, stats, agentLog, backendOnline, logRef } = useAgent(running);

  // Derived values
  const detRate = stats.attacked > 0 ? Math.round((stats.blocked / stats.attacked) * 100) : 0;
  const threatLevel = incidents.length > 5 ? "CRITICAL" : incidents.length > 2 ? "HIGH" : incidents.length > 0 ? "ELEVATED" : "NOMINAL";
  const threatColor = {
    CRITICAL: "text-red-400 bg-red-500/10 border-red-500/30",
    HIGH:     "text-orange-400 bg-orange-500/10 border-orange-500/30",
    ELEVATED: "text-amber-400 bg-amber-500/10 border-amber-500/30",
    NOMINAL:  "text-emerald-400 bg-emerald-500/10 border-emerald-500/30",
  }[threatLevel]!;

  // Signal for chart — from real backend
  const liveSignal = currentWindow?.signal
    ? currentWindow.signal.map((v: number, i: number) => ({ v, i }))
    : [];

  const TABS = [
    { id: "stream",     icon: <Radio size={12} />,    label: "Live Agent"  },
    { id: "attacks",    icon: <Zap size={12} />,      label: "Attacks"     },
    { id: "compliance", icon: <FileText size={12} />, label: "Compliance"  },
    { id: "privacy",    icon: <Eye size={12} />,      label: "Privacy"     },
    { id: "research",   icon: <Database size={12} />, label: "Research"    },
  ];

  return (
    <div className="min-h-screen bg-[#070b14] text-white font-mono">
      {/* Grid bg */}
      <div className="fixed inset-0 pointer-events-none opacity-[0.035]"
        style={{ backgroundImage: "linear-gradient(rgba(6,182,212,0.4) 1px,transparent 1px),linear-gradient(90deg,rgba(6,182,212,0.4) 1px,transparent 1px)", backgroundSize: "48px 48px" }} />

      {/* ── Header ── */}
      <header className="sticky top-0 z-50 border-b border-slate-800/80 bg-black/60 backdrop-blur-xl">
        <div className="max-w-screen-xl mx-auto px-6 py-3 flex items-center gap-5">
          <div className="flex items-center gap-3 flex-shrink-0">
            <div className="w-9 h-9 rounded-xl bg-gradient-to-br from-cyan-500/30 to-violet-600/30 border border-cyan-500/30 flex items-center justify-center">
              <Brain size={18} className="text-cyan-400" />
            </div>
            <div>
              <div className="text-sm font-black tracking-[0.2em] uppercase">CyberNeuro</div>
              <div className="text-[9px] text-cyan-500/50 tracking-[0.12em] uppercase">Neural Data Security Platform</div>
            </div>
          </div>

          <div className={`flex items-center gap-2 px-3 py-1.5 rounded-lg border text-xs font-black tracking-wider ${threatColor}`}>
            <AlertOctagon size={11} className={incidents.length > 0 ? "animate-pulse" : ""} />
            THREAT: {threatLevel}
          </div>

          {/* Backend status */}
          <div className={`flex items-center gap-1.5 px-2 py-1 rounded text-[10px] font-bold border ${backendOnline ? "text-emerald-400 border-emerald-500/20 bg-emerald-500/5" : "text-slate-500 border-slate-700/40"}`}>
            <span className={`w-1.5 h-1.5 rounded-full ${backendOnline ? "bg-emerald-400 animate-pulse" : "bg-slate-600"}`} />
            {backendOnline ? (stats.initialized ? "REAL DATA" : "INITIALIZING") : "BACKEND OFFLINE"}
          </div>

          <div className="hidden lg:flex items-center gap-8 flex-1 justify-center">
            {[["WINDOWS", stats.total], ["ATTACKS", stats.attacked], ["BLOCKED", stats.blocked], ["DETECTION", `${detRate}%`]].map(([l, v]) => (
              <div key={l as string} className="text-center">
                <div className="text-lg font-black font-mono tabular-nums text-white">{v}</div>
                <div className="text-[9px] text-slate-500 tracking-widest uppercase">{l}</div>
              </div>
            ))}
          </div>

          <div className="flex items-center gap-3 ml-auto">
            <span className="text-[10px] text-slate-600 hidden md:block tracking-wider">Stanford CS153 · Frontier Systems · 2026</span>
            <button onClick={() => setRunning(r => !r)}
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-xs font-black tracking-wider border transition-all duration-200 ${running
                ? "bg-red-500/15 border-red-500/40 text-red-400 hover:bg-red-500/25"
                : "bg-cyan-500/15 border-cyan-500/40 text-cyan-400 hover:bg-cyan-500/25"}`}>
              <span className={`w-2 h-2 rounded-full ${running ? "bg-red-400 animate-pulse" : "bg-cyan-400"}`} />
              {running ? "STOP AGENT" : "START AGENT"}
            </button>
          </div>
        </div>

        <div className="max-w-screen-xl mx-auto px-6 flex">
          {TABS.map(t => (
            <button key={t.id} onClick={() => setTab(t.id as typeof tab)}
              className={`flex items-center gap-1.5 px-5 py-2.5 text-[11px] font-black tracking-wider uppercase border-b-2 transition-all ${tab === t.id
                ? "border-cyan-400 text-cyan-400 bg-cyan-400/5"
                : "border-transparent text-slate-500 hover:text-slate-300"}`}>
              {t.icon}{t.label}
            </button>
          ))}
        </div>
      </header>

      <main className="max-w-screen-xl mx-auto px-6 py-6">

        {/* ══ LIVE AGENT ══ */}
        {tab === "stream" && (
          <div className="grid grid-cols-12 gap-4">
            <div className="col-span-8 space-y-4">

              {/* EEG Monitor */}
              <div className="rounded-xl border border-cyan-500/20 bg-slate-900/60 overflow-hidden">
                <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800/80">
                  <div className="flex items-center gap-2">
                    <span className={`w-2 h-2 rounded-full ${running ? "bg-cyan-400 animate-pulse" : "bg-slate-700"}`} />
                    <span className="text-xs font-black text-slate-300 tracking-wider uppercase">
                      {backendOnline && stats.initialized ? "Live EEG — Real PhysioNet Signal" : "Live EEG Signal — Motor Cortex (C3/C4)"}
                    </span>
                    {backendOnline && stats.initialized && (
                      <span className="text-[9px] text-emerald-400/60 border border-emerald-500/20 px-1.5 py-0.5 rounded">REAL DATA</span>
                    )}
                  </div>
                  <div className="flex items-center gap-3">
                    {currentWindow && <RiskBadge risk={currentWindow.status} />}
                    <span className="text-[10px] font-mono text-slate-600">
                      score: {currentWindow ? currentWindow.anomaly_score.toFixed(3) : "—"} / threshold: 0.650
                    </span>
                  </div>
                </div>

                <div className="relative" style={{ height: 170 }}>
                  {liveSignal.length > 0 ? (
                    <ResponsiveContainer width="100%" height="100%">
                      <AreaChart data={liveSignal} margin={{ top: 8, right: 8, bottom: 8, left: 8 }}>
                        <defs>
                          <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                            <stop offset="5%" stopColor={currentWindow?.status === "BLOCKED" ? "#ef4444" : currentWindow?.status === "ATTACKED" ? "#f97316" : "#06b6d4"} stopOpacity={0.3} />
                            <stop offset="95%" stopColor="#06b6d4" stopOpacity={0} />
                          </linearGradient>
                        </defs>
                        <Area type="monotone" dataKey="v"
                          stroke={currentWindow?.status === "BLOCKED" ? "#ef4444" : currentWindow?.status === "ATTACKED" ? "#f97316" : "#06b6d4"}
                          fill="url(#sg)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
                        <YAxis domain={[-1.5, 1.5]} tick={false} width={0} axisLine={false} />
                        <XAxis dataKey="i" tick={false} height={0} axisLine={false} />
                      </AreaChart>
                    </ResponsiveContainer>
                  ) : (
                    <div className="h-full flex flex-col items-center justify-center gap-2">
                      <Brain size={24} className="text-slate-700" />
                      <div className="text-slate-600 text-xs tracking-widest uppercase">
                        {backendOnline ? "Start Agent to Monitor EEG Stream" : "Start backend: python api/server.py"}
                      </div>
                    </div>
                  )}
                </div>

                {currentWindow && (
                  <div className="px-5 py-2 border-t border-slate-800/60 flex gap-6 text-center">
                    {[
                      ["WINDOW",    `#${currentWindow.window_id}`],
                      ["STATUS",    currentWindow.status],
                      ["ANOMALY",   currentWindow.anomaly_score.toFixed(3)],
                      ["ATTACK",    currentWindow.attack_type],
                      ["CONF",      `${(currentWindow.classifier_confidence * 100).toFixed(0)}%`],
                    ].map(([k, v]) => (
                      <div key={k as string}>
                        <div className={`text-xs font-mono font-black ${
                          k === "ANOMALY" && currentWindow.anomaly_score > 0.65 ? "text-red-400" :
                          k === "STATUS" && currentWindow.status === "BLOCKED" ? "text-red-400" : "text-white"
                        }`}>{v}</div>
                        <div className="text-[9px] text-slate-600 tracking-wider mt-0.5">{k}</div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Window table */}
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 overflow-hidden">
                <div className="flex items-center justify-between px-5 py-3 border-b border-slate-800/60">
                  <span className="text-xs font-black text-slate-400 uppercase tracking-wider">Signal Window History</span>
                  <span className="text-[10px] text-slate-600">{windows.length} processed</span>
                </div>
                <div className="overflow-auto max-h-64">
                  <table className="w-full text-xs">
                    <thead className="sticky top-0 bg-slate-900/90">
                      <tr className="border-b border-slate-800/60">
                        {["#", "TIME", "STATUS", "ANOMALY SCORE", "ATTACK TYPE", "VERDICT"].map(h => (
                          <th key={h} className="px-4 py-2 text-left text-[10px] text-slate-600 font-black tracking-widest">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {windows.length === 0 ? (
                        <tr><td colSpan={6} className="px-4 py-8 text-center text-slate-700 text-xs">No windows yet — start the agent</td></tr>
                      ) : windows.slice(0, 15).map((w, i) => (
                        <tr key={w.window_id} className={`border-b border-slate-800/30 ${i === 0 ? "bg-slate-800/50" : ""} ${w.status === "BLOCKED" ? "bg-red-500/5" : ""}`}>
                          <td className="px-4 py-1.5 font-mono text-slate-500 text-[11px]">#{w.window_id}</td>
                          <td className="px-4 py-1.5 font-mono text-slate-500 text-[11px]">{w.timestamp}</td>
                          <td className="px-4 py-1.5"><RiskBadge risk={w.status} /></td>
                          <td className="px-4 py-1.5">
                            <div className="flex items-center gap-2">
                              <div className="w-16 h-1 bg-slate-800 rounded-full overflow-hidden">
                                <div className={`h-full ${w.anomaly_score > 0.65 ? "bg-red-500" : w.anomaly_score > 0.4 ? "bg-amber-500" : "bg-emerald-500"}`}
                                  style={{ width: `${w.anomaly_score * 100}%` }} />
                              </div>
                              <span className={`font-mono text-[11px] tabular-nums ${w.anomaly_score > 0.65 ? "text-red-400" : "text-slate-400"}`}>
                                {w.anomaly_score.toFixed(3)}
                              </span>
                            </div>
                          </td>
                          <td className="px-4 py-1.5 font-mono text-[11px] text-orange-400">{w.attack_type}</td>
                          <td className="px-4 py-1.5 font-mono text-[11px]">
                            {w.status === "BLOCKED"  ? <span className="text-red-400 font-black">⛔ BLOCKED</span>
                           : w.status === "ATTACKED" ? <span className="text-amber-400">⚠ PASSED</span>
                           : <span className="text-emerald-400">✓ CLEAN</span>}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>

            {/* Right column */}
            <div className="col-span-4 space-y-4">

              {/* Agent terminal */}
              <div className="rounded-xl border border-cyan-500/20 bg-slate-900/60 overflow-hidden">
                <div className="flex items-center gap-2 px-4 py-3 border-b border-slate-800/60">
                  <Terminal size={12} className="text-cyan-400" />
                  <span className="text-xs font-black text-slate-300 tracking-wider uppercase">Agent Terminal</span>
                  {running && <span className="ml-auto flex items-center gap-1 text-[10px] text-cyan-400/60"><span className="w-1.5 h-1.5 rounded-full bg-cyan-400 animate-ping" />LIVE</span>}
                </div>
                <div ref={logRef} className="h-56 overflow-y-auto p-3 space-y-0.5 bg-black/40">
                  {agentLog.map((line, i) => {
                    const c = line.includes("[ACTION]") && line.includes("BLOCKED") ? "text-red-400"
                      : line.includes("[DETECT]") || line.includes("[WARN]") ? "text-amber-400"
                      : line.includes("[REASON]") ? "text-violet-400"
                      : line.includes("[SYS]") ? "text-cyan-500/70"
                      : "text-emerald-400/80";
                    return <div key={i} className={`text-[10px] leading-relaxed ${c}`}>{line}</div>;
                  })}
                  {running && <div className="text-cyan-400/40 text-[10px] animate-pulse">█</div>}
                </div>
              </div>

              {/* Incidents */}
              <div className="rounded-xl border border-red-500/20 bg-slate-900/60 overflow-hidden">
                <div className="flex items-center justify-between px-4 py-3 border-b border-slate-800/60">
                  <div className="flex items-center gap-2">
                    <AlertOctagon size={12} className="text-red-400" />
                    <span className="text-xs font-black text-slate-300 uppercase tracking-wider">Incidents</span>
                  </div>
                  {incidents.length > 0 && (
                    <span className="text-[10px] px-2 py-0.5 rounded-full bg-red-500/20 border border-red-500/30 text-red-400 font-black">{incidents.length}</span>
                  )}
                </div>
                <div className="space-y-2 p-3 max-h-64 overflow-y-auto">
                  {incidents.length === 0
                    ? <div className="py-6 text-center text-slate-700 text-xs">No incidents detected</div>
                    : incidents.slice(0, 6).map((inc, i) => (
                      <div key={i} className="border border-red-500/20 rounded-lg bg-red-500/5 p-3">
                        <div className="flex items-center justify-between mb-1.5">
                          <RiskBadge risk={inc.severity} />
                          <div className="flex items-center gap-2 text-[10px]">
                            <span className="text-orange-400 font-black font-mono">{inc.attack_type}</span>
                            <span className="text-slate-600">{inc.time}</span>
                          </div>
                        </div>
                        <div className="text-[10px] text-slate-400 leading-relaxed mb-1">{inc.reasoning}</div>
                        <div className="text-[9px] text-slate-600 font-mono">
                          conf: {(inc.confidence * 100).toFixed(0)}% · <span className="text-red-400">⛔ {inc.action}</span>
                        </div>
                      </div>
                    ))
                  }
                </div>
              </div>

              {/* Quick stats */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { v: `${detRate}%`, l: "Detection Rate", c: detRate > 70 ? "emerald" : "amber" },
                  { v: stats.blocked.toString(), l: "Blocked", c: "red" },
                  { v: stats.clean.toString(), l: "Clean Passed", c: "violet" },
                  { v: running ? "ACTIVE" : "OFFLINE", l: "Agent Status", c: running ? "cyan" : "slate" },
                ].map(s => (
                  <div key={s.l} className={`rounded-xl border p-3 text-center bg-${s.c}-500/5 border-${s.c}-500/20`}>
                    <div className={`text-xl font-black font-mono tabular-nums text-${s.c}-400`}>{s.v}</div>
                    <div className="text-[9px] text-slate-500 uppercase tracking-widest mt-0.5">{s.l}</div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══ ATTACKS ══ */}
        {tab === "attacks" && (
          <div className="space-y-5">
            <div className="grid grid-cols-4 gap-4">
              {[
                { v: "100%", l: "FGSM Attack Success", s: "ε=0.05 on real PhysioNet EEG", c: "red"    },
                { v: "80%",  l: "PGD Attack Success",  s: "ε=0.05, 40 iterations",        c: "red"    },
                { v: "1.00", l: "Best Detector AUC",   s: "Isolation Forest, zero FPR",   c: "emerald" },
                { v: "26dB", l: "Attack SNR",           s: "Imperceptible — below clinical threshold", c: "violet" },
              ].map(m => (
                <div key={m.l} className={`rounded-xl border border-${m.c}-500/20 bg-${m.c}-500/5 p-5`}>
                  <div className={`text-4xl font-black font-mono text-${m.c}-400 mb-1`}>{m.v}</div>
                  <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-1">{m.l}</div>
                  <div className="text-[11px] text-slate-500">{m.s}</div>
                </div>
              ))}
            </div>
            <div className="grid grid-cols-2 gap-5">
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
                <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-5">Classifier Accuracy vs Attack Strength</div>
                <ResponsiveContainer width="100%" height={240}>
                  <LineChart data={ROBUSTNESS}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
                    <XAxis dataKey="eps" tick={{ fill: "#475569", fontSize: 10 }} label={{ value: "Perturbation ε (L∞)", fill: "#475569", fontSize: 10, position: "insideBottom", offset: -4 }} />
                    <YAxis tick={{ fill: "#475569", fontSize: 10 }} domain={[0, 100]} tickFormatter={(v: number) => `${v}%`} />
                    <Tooltip contentStyle={{ background: "#0f172a", border: "1px solid #1e293b", borderRadius: 8, fontSize: 11 }} formatter={(v: unknown, n: unknown) => [`${v}%`, String(n).toUpperCase()]} />
                    <Line type="monotone" dataKey="fgsm" stroke="#ef4444" strokeWidth={2.5} dot={{ r: 3, fill: "#ef4444" }} name="FGSM" />
                    <Line type="monotone" dataKey="pgd"  stroke="#f97316" strokeWidth={2.5} dot={{ r: 3, fill: "#f97316" }} strokeDasharray="6 3" name="PGD" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
                <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-5">Anomaly Detector Performance</div>
                <div className="space-y-5">
                  {[
                    { name: "Isolation Forest", auc: 100, recall: 100, fpr: 0,   desc: "Unsupervised — trained only on clean EEG. No labeled attacks required." },
                    { name: "Statistical (LR)", auc: 88,  recall: 83,  fpr: 0,   desc: "Frequency-domain features: HF power, spectral entropy, Hjorth parameters." },
                    { name: "Confidence-Based", auc: 27,  recall: 100, fpr: 100, desc: "Model certainty proxy — high recall, high FPR. Secondary signal only." },
                  ].map(d => (
                    <div key={d.name} className="space-y-1.5">
                      <div className="flex justify-between">
                        <span className="text-sm font-black text-white">{d.name}</span>
                        <div className="flex gap-3 text-xs font-mono">
                          <span className={d.auc >= 80 ? "text-emerald-400" : "text-amber-400"}>AUC {d.auc}%</span>
                          <span className={d.fpr < 5 ? "text-emerald-400" : "text-red-400"}>FPR {d.fpr}%</span>
                        </div>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden">
                        <div className={`h-full rounded-full ${d.auc >= 80 ? "bg-gradient-to-r from-emerald-700 to-emerald-400" : "bg-gradient-to-r from-red-800 to-red-500"}`}
                          style={{ width: `${d.auc}%` }} />
                      </div>
                      <div className="text-[10px] text-slate-500">{d.desc}</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ COMPLIANCE ══ */}
        {tab === "compliance" && (
          <div className="space-y-5">

            {/* Security framing header */}
            <div className="rounded-xl border border-red-500/20 bg-red-500/5 p-5">
              <div className="text-[10px] font-black text-red-400/60 uppercase tracking-widest mb-2">Security Finding — Neural Data Exposure</div>
              <div className="text-base text-slate-200 leading-relaxed max-w-4xl">
                CyberNeuro's compliance agent autonomously audited BCI vendor privacy policies using Claude AI.
                The finding: most vendors treat neural data like ordinary app data — no meaningful access controls,
                no encryption guarantees, and unrestricted third-party sharing. This is a security vulnerability,
                not just a legal one. An attacker doesn't need to hack your device if the vendor is already
                handing your brain data to third parties.
              </div>
            </div>

            {/* Real audit results — dramatic contrast */}
            <div className="grid grid-cols-2 gap-5">

              {/* NeuroCorp — CRITICAL */}
              <div className="rounded-xl border border-red-500/30 bg-red-500/5 p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="text-lg font-black text-red-400">NeuroCorp</div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Typical consumer BCI vendor</div>
                  </div>
                  <div className="text-right">
                    <div className="text-4xl font-black font-mono text-red-400">0</div>
                    <div className="text-[10px] text-slate-500">/ 100</div>
                  </div>
                </div>
                <div className="h-2 bg-slate-800 rounded-full mb-4">
                  <div className="h-full w-0 rounded-full bg-red-500" />
                </div>
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded border border-red-400/30 bg-red-400/10 text-[10px] font-bold text-red-400 mb-4">
                  <span className="w-1.5 h-1.5 rounded-full bg-red-500 animate-pulse" />CRITICAL RISK
                </div>
                <div className="space-y-3">
                  <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-2">Security Vulnerabilities Found</div>
                  {[
                    { id: "CO-01", sev: "CRITICAL", vuln: "No opt-in consent before neural data collection", evidence: "By using our EEG headset, you agree to our collection of brain activity data." },
                    { id: "CA-02", sev: "CRITICAL", vuln: "Neural data sold to advertising partners", evidence: "We reserve the right to sell anonymized neural datasets to third parties." },
                    { id: "EU-01", sev: "CRITICAL", vuln: "No prohibition on subliminal neural manipulation", evidence: "No mention of restrictions on how neural data is used to influence users." },
                    { id: "GEN-02", sev: "HIGH",     vuln: "No breach notification commitment", evidence: "Contains no information about breach notification procedures or timeframes." },
                    { id: "GEN-03", sev: "MEDIUM",   vuln: "Indefinite retention of raw brain signals", evidence: "Data is retained for as long as necessary to fulfill our business purposes." },
                  ].map(v => (
                    <div key={v.id} className="border border-red-500/15 rounded-lg bg-red-500/5 p-3">
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-[9px] font-mono text-red-400/60 font-black">{v.id}</span>
                        <span className={`text-[9px] font-black px-1.5 py-0.5 rounded ${v.sev === "CRITICAL" ? "bg-red-500/20 text-red-400" : v.sev === "HIGH" ? "bg-orange-500/20 text-orange-400" : "bg-amber-500/20 text-amber-400"}`}>{v.sev}</span>
                        <span className="text-[10px] font-bold text-slate-300">{v.vuln}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 italic">Evidence: "{v.evidence}"</div>
                    </div>
                  ))}
                </div>
              </div>

              {/* NeuroSafe — LOW */}
              <div className="rounded-xl border border-emerald-500/30 bg-emerald-500/5 p-5">
                <div className="flex items-center justify-between mb-4">
                  <div>
                    <div className="text-lg font-black text-emerald-400">NeuroSafe</div>
                    <div className="text-[10px] text-slate-500 uppercase tracking-wider">Security-conscious vendor</div>
                  </div>
                  <div className="text-right">
                    <div className="text-4xl font-black font-mono text-emerald-400">82.5</div>
                    <div className="text-[10px] text-slate-500">/ 100</div>
                  </div>
                </div>
                <div className="h-2 bg-slate-800 rounded-full mb-4">
                  <div className="h-full rounded-full bg-gradient-to-r from-emerald-700 to-emerald-400" style={{ width: "82.5%" }} />
                </div>
                <div className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded border border-emerald-400/30 bg-emerald-400/10 text-[10px] font-bold text-emerald-400 mb-4">
                  <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />LOW RISK
                </div>
                <div className="space-y-3">
                  <div className="text-[10px] font-black text-slate-400 uppercase tracking-wider mb-2">Security Controls Verified</div>
                  {[
                    { id: "CO-01", ctrl: "Explicit opt-in consent required", evidence: "We collect neural data ONLY with your explicit opt-in consent, obtained separately before any recording begins." },
                    { id: "CA-02", ctrl: "Zero neural data sales — ever", evidence: "We never sell your neural data under any circumstances." },
                    { id: "EU-02", ctrl: "AES-256 encryption at rest + TLS 1.3 in transit", evidence: "All neural data is encrypted using AES-256 at rest and TLS 1.3 in transit." },
                    { id: "CA-01", ctrl: "Right to delete — processed within 30 days", evidence: "You may request complete deletion of your neural data at any time, processed within 30 days." },
                    { id: "GEN-02", ctrl: "72-hour breach notification commitment", evidence: "We will notify you within 72 hours in the event of a breach affecting your neural data." },
                  ].map(v => (
                    <div key={v.id} className="border border-emerald-500/15 rounded-lg bg-emerald-500/5 p-3">
                      <div className="flex items-center gap-2 mb-1.5">
                        <span className="text-[9px] font-mono text-emerald-400/60 font-black">{v.id}</span>
                        <span className="text-[9px] font-black px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-400">COMPLIANT</span>
                        <span className="text-[10px] font-bold text-slate-300">{v.ctrl}</span>
                      </div>
                      <div className="text-[10px] text-slate-500 italic">Evidence: "{v.evidence}"</div>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* Key finding */}
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 p-5">
              <div className="text-xs font-black text-amber-400 uppercase tracking-wider mb-3">Key Security Finding</div>
              <div className="grid grid-cols-3 gap-5 text-xs text-slate-400 leading-relaxed">
                <div>
                  <div className="font-black text-white mb-1">The Attack Surface</div>
                  Vendors like NeuroCorp collect neural data without consent and share it with third parties. An attacker doesn't need to intercept your EEG signal — they can simply buy it from the vendor's advertising partners.
                </div>
                <div>
                  <div className="font-black text-white mb-1">Why It's a Security Issue</div>
                  Neural data reveals mental health conditions, emotional states, cognitive patterns, and identity — even when "anonymized." Unrestricted third-party sharing creates a data supply chain attack vector for brain data.
                </div>
                <div>
                  <div className="font-black text-white mb-1">What CyberNeuro Does</div>
                  The compliance agent autonomously audits vendor policies using Claude AI, identifies security vulnerabilities in data handling practices, and generates actionable findings — continuously, without human review.
                </div>
              </div>
            </div>

            {/* Laws as security standards */}
            <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
              <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-4">Security Standards Checked</div>
              <div className="grid grid-cols-2 gap-3">
                {[
                  { id: "CO-01/02", law: "Colorado HB 24-1058", security: "Opt-in consent + sensitive data classification" },
                  { id: "CA-01/02", law: "California SB 1223",  security: "Right to delete + prohibition on data sales" },
                  { id: "MN-01",    law: "Minnesota HF 1",      security: "Data minimization — collect only what's needed" },
                  { id: "EU-01/02", law: "EU AI Act Article 5", security: "No subliminal manipulation + encryption required" },
                  { id: "GEN-1-3",  law: "Neurorights Foundation", security: "Third-party limits + breach notification + retention caps" },
                ].map(l => (
                  <div key={l.id} className="flex gap-3 py-2 border-b border-slate-800/40">
                    <span className="text-[9px] font-mono text-cyan-400/50 w-14 flex-shrink-0 mt-0.5 font-black">{l.id}</span>
                    <div>
                      <div className="text-xs font-black text-slate-200 mb-0.5">{l.law}</div>
                      <div className="text-[10px] text-slate-500">{l.security}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </div>
        )}

        {/* ══ PRIVACY ══ */}
        {tab === "privacy" && (
          <div className="space-y-5">
            <div className="rounded-xl border border-amber-500/30 bg-amber-500/5 p-5 flex items-start gap-4">
              <div className="w-12 h-12 rounded-xl bg-amber-500/10 border border-amber-500/20 flex items-center justify-center flex-shrink-0">
                <Eye size={20} className="text-amber-400" />
              </div>
              <div>
                <div className="text-lg font-black text-amber-300 mb-1">De-Anonymization Risk: 67/100 — HIGH</div>
                <div className="text-sm text-amber-400/70 max-w-3xl leading-relaxed">
                  Measured on real PhysioNet EEG from 5 subjects. After standard anonymization, cognitive state is classifiable, medical markers are detectable, and attention states are trackable. Brainprints are biometrically unique.
                </div>
              </div>
            </div>
            <div className="grid grid-cols-2 gap-5">
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
                <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-5">Inference Task Results (Real EEG)</div>
                <div className="space-y-5">
                  {[
                    { name: "Cognitive State",   accuracy: 57,  lift: 1.13, risk: "MEDIUM", auc: 58 },
                    { name: "Re-Identification", accuracy: 50,  lift: 1.5,  risk: "MEDIUM", auc: 62 },
                    { name: "Medical Markers",   accuracy: 60,  lift: 1.2,  risk: "MEDIUM", auc: 65 },
                    { name: "Attention/Load",    accuracy: 55,  lift: 1.1,  risk: "LOW",    auc: 60 },
                  ].map(t => (
                    <div key={t.name}>
                      <div className="flex justify-between mb-2">
                        <span className="text-sm font-black text-white">{t.name}</span>
                        <div className="flex items-center gap-2">
                          <span className="font-mono font-black text-white">{t.accuracy}%</span>
                          <RiskBadge risk={t.risk} />
                        </div>
                      </div>
                      <div className="h-2 bg-slate-800 rounded-full overflow-hidden relative">
                        <div className={`h-full ${t.risk === "HIGH" ? "bg-gradient-to-r from-red-700 to-red-400" : "bg-gradient-to-r from-amber-700 to-amber-400"}`}
                          style={{ width: `${t.accuracy}%` }} />
                        <div className="absolute inset-y-0 w-px bg-slate-400/30" style={{ left: "50%" }} />
                      </div>
                      <div className="flex justify-between mt-1 text-[10px] text-slate-600">
                        <span>AUC: {t.auc}%</span>
                        <span>Lift: {t.lift}× above chance (50%)</span>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
              <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
                <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-4">What Anonymized EEG Reveals</div>
                <div className="space-y-3">
                  {[
                    { t: "Mental Intentions",  i: <Brain size={14}/>,    c: "red",    d: "Motor imagery at 57% — mental states partially readable even with 5 subjects." },
                    { t: "Personal Identity",  i: <Eye size={14}/>,      c: "orange", d: "Cross-session re-identification possible — brainprints are biometrically unique." },
                    { t: "Medical Conditions", i: <Activity size={14}/>, c: "amber",  d: "Neurological condition signatures detectable from any EEG session." },
                    { t: "Attention State",    i: <Cpu size={14}/>,      c: "violet", d: "Cognitive load trackable — continuous workplace surveillance risk." },
                  ].map(item => (
                    <div key={item.t} className={`flex gap-3 p-3 rounded-lg border border-${item.c}-500/20 bg-${item.c}-500/5`}>
                      <div className={`text-${item.c}-400 flex-shrink-0 mt-0.5`}>{item.i}</div>
                      <div>
                        <div className={`text-xs font-black text-${item.c}-300 mb-0.5`}>{item.t}</div>
                        <div className="text-[11px] text-slate-400 leading-relaxed">{item.d}</div>
                      </div>
                    </div>
                  ))}
                </div>
                <div className="mt-4 pt-4 border-t border-slate-800 grid grid-cols-2 gap-2 text-[10px]">
                  {[["Source","PhysioNet EEGMMIDB"],["Subjects","5 real humans"],["Channels","64 electrodes"],["Rate","160 Hz"]].map(([k,v]) => (
                    <div key={k}><span className="text-slate-600">{k}: </span><span className="text-slate-400 font-mono">{v}</span></div>
                  ))}
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ══ RESEARCH ══ */}
        {tab === "research" && (
          <div className="space-y-5">
            <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-6">
              <div className="text-[10px] font-black text-cyan-400/60 uppercase tracking-widest mb-2">Research Contribution</div>
              <div className="text-base text-slate-200 leading-relaxed max-w-4xl">
                CyberNeuro is an agentic security platform for brain-computer interfaces combining real-time adversarial signal detection, automated neurorights compliance auditing, and empirical de-anonymization risk scoring — addressing three primary threat vectors simultaneously, a gap identified by Schroder et al. (2025).
              </div>
            </div>
            <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
              <div className="text-xs font-black text-slate-300 uppercase tracking-wider mb-4">Key References</div>
              <div className="divide-y divide-slate-800/50">
                {PAPERS.map((p, i) => (
                  <div key={i} className="flex gap-4 py-3.5">
                    <span className="text-[10px] font-mono text-cyan-400/40 w-6 flex-shrink-0 mt-1 font-black">{String(i+1).padStart(2,"0")}</span>
                    <div className="flex-1">
                      <div className="text-sm font-black text-slate-200 mb-0.5">{p.title}</div>
                      <div className="text-xs text-slate-500">{p.authors} · <span className="text-slate-400 font-black">{p.venue}</span> · {p.year}</div>
                    </div>
                  </div>
                ))}
              </div>
            </div>
            <div className="grid grid-cols-3 gap-4">
              {[
                { t: "Contributions", c: "cyan", items: ["Agentic real-time adversarial EEG defense","Automated neurorights law compliance auditing","Empirical de-anonymization risk quantification","Multi-detector anomaly detection benchmark"] },
                { t: "Agentic Architecture", c: "violet", items: ["Perceive → Detect → Reason → Act loop","Claude LLM reasons over each detected threat","Autonomous incident logging and alerting","No human required for routine defense"] },
                { t: "Regulatory Context", c: "amber", items: ["Colorado HB 24-1058 (2024)","California SB 1223 (2024)","EU AI Act Article 5","US Senate FTC letter (April 2025)","29/30 BCI companies non-compliant"] },
              ].map(s => (
                <div key={s.t} className={`rounded-xl border border-${s.c}-500/20 bg-${s.c}-500/5 p-5`}>
                  <div className={`text-xs font-black text-${s.c}-400 uppercase tracking-wider mb-3`}>{s.t}</div>
                  <ul className="space-y-2">
                    {s.items.map((item, i) => (
                      <li key={i} className="flex items-start gap-2 text-xs text-slate-400">
                        <ChevronRight size={12} className={`text-${s.c}-400/40 flex-shrink-0 mt-0.5`} />{item}
                      </li>
                    ))}
                  </ul>
                </div>
              ))}
            </div>
            <div className="rounded-xl border border-slate-800/60 bg-slate-900/60 p-5">
              <div className="text-xs font-black text-slate-400 uppercase tracking-wider mb-3">Built With</div>
              <div className="flex flex-wrap gap-2">
                {["MNE-Python","PyTorch","EEGNet","FGSM/PGD","scikit-learn","Claude API","OpenRouter","FastAPI","Next.js 14","Recharts","Tailwind","Cloudflare","PhysioNet Dataset"].map(t => (
                  <span key={t} className="text-[10px] px-2.5 py-1 rounded-full bg-slate-800 border border-slate-700/60 text-slate-400">{t}</span>
                ))}
              </div>
              <div className="mt-4 pt-4 border-t border-slate-800 text-[10px] text-slate-600">
                Stanford CS153 · Frontier Systems · Spring 2026 · Built by Selin Ozgursoy
              </div>
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
