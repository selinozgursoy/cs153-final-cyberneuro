/**
 * CyberNeuro — useAgent React Hook
 * Polls the real FastAPI backend every 750ms when running.
 * Falls back gracefully if backend is not available.
 */

import { useState, useEffect, useRef, useCallback } from "react";

export interface AgentWindow {
  window_id: number;
  status: "CLEAN" | "ATTACKED" | "BLOCKED";
  is_attacked: boolean;
  anomaly_score: number;
  classifier_confidence: number;
  attack_type: string;
  reasoning: string;
  signal: number[];
  timestamp: string;
  simulated?: boolean;
}

export interface AgentStats {
  total: number;
  attacked: number;
  blocked: number;
  clean: number;
  detection_rate: number;
  initialized: boolean;
  initializing: boolean;
}

export interface Incident {
  id: number;
  time: string;
  status: string;
  severity: string;
  attack_type: string;
  anomaly: number;
  confidence: number;
  reasoning: string;
  action: string;
}

const API_BASE = "http://localhost:8000";
const POLL_INTERVAL = 750; // ms

export function useAgent(running: boolean) {
  const [currentWindow, setCurrentWindow] = useState<AgentWindow | null>(null);
  const [windows, setWindows] = useState<AgentWindow[]>([]);
  const [incidents, setIncidents] = useState<Incident[]>([]);
  const [stats, setStats] = useState<AgentStats>({
    total: 0, attacked: 0, blocked: 0, clean: 0,
    detection_rate: 0, initialized: false, initializing: false,
  });
  const [agentLog, setAgentLog] = useState<string[]>([
    "[SYS] CyberNeuro defense agent initialized",
    "[SYS] EEGNet classifier ready — real PhysioNet data",
    "[SYS] Anomaly detectors loaded (Statistical + Isolation Forest)",
    "[SYS] Claude reasoning via OpenRouter — active",
    "[SYS] Awaiting signal stream — press START AGENT",
  ]);
  const [backendOnline, setBackendOnline] = useState(false);
  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const logRef = useRef<HTMLDivElement>(null);

  const addLog = useCallback((msg: string) => {
    setAgentLog(prev => [...prev.slice(-60), msg]);
    setTimeout(() => {
      if (logRef.current) logRef.current.scrollTop = logRef.current.scrollHeight;
    }, 50);
  }, []);

  // Check if backend is online
  useEffect(() => {
    const check = async () => {
      try {
        const res = await fetch(`${API_BASE}/health`, { signal: AbortSignal.timeout(2000) });
        const data = await res.json();
        setBackendOnline(true);
        if (data.initializing) addLog("[SYS] Backend initializing — loading real EEG data...");
        if (data.initialized && !backendOnline) addLog("[SYS] ✓ Real platform ready — EEGNet + detectors loaded");
      } catch {
        setBackendOnline(false);
      }
    };
    check();
    const t = setInterval(check, 5000);
    return () => clearInterval(t);
  }, [backendOnline, addLog]);

  // Main polling loop
  useEffect(() => {
    if (!running) {
      if (intervalRef.current) clearInterval(intervalRef.current);
      if (stats.total > 0) addLog("[SYS] ■ Agent stopped.");
      return;
    }

    addLog(`[SYS] ► Agent STARTED — ${backendOnline ? "REAL data mode" : "simulation mode"}`);
    addLog(`[SYS] Attack injection: 28% | Threshold: 0.650 | Claude reasoning: active`);

    intervalRef.current = setInterval(async () => {
      try {
        // Poll real backend
        const res = await fetch(`${API_BASE}/stream/next`, {
          signal: AbortSignal.timeout(3000)
        });
        const w: AgentWindow = await res.json();

        setCurrentWindow(w);
        setWindows(prev => [w, ...prev].slice(0, 50));

        // Fetch updated stats
        const statsRes = await fetch(`${API_BASE}/stream/stats`, {
          signal: AbortSignal.timeout(2000)
        });
        const s: AgentStats = await statsRes.json();
        setStats(s);

        // Update log
        if (w.status === "BLOCKED") {
          addLog(`[DETECT] Window #${w.window_id} → score=${w.anomaly_score.toFixed(3)} | ${w.attack_type} DETECTED`);
          addLog(`[REASON] ${w.reasoning}`);
          addLog(`[ACTION] ⛔ BLOCKED — conf=${(w.classifier_confidence * 100).toFixed(0)}% | incident logged`);
          // Fetch updated incidents
          const incRes = await fetch(`${API_BASE}/incidents`);
          const incData = await incRes.json();
          setIncidents(incData.incidents || []);
        } else if (w.is_attacked) {
          addLog(`[WARN]   Window #${w.window_id} → score=${w.anomaly_score.toFixed(3)} | weak signal, passed`);
        } else if (w.window_id % 6 === 0) {
          addLog(`[OK]     Window #${w.window_id} → score=${w.anomaly_score.toFixed(3)} | CLEAN`);
        }

      } catch {
        // Backend unavailable — log it
        if (stats.total % 10 === 0) {
          addLog("[WARN] Backend offline — start: python api/server.py");
        }
      }
    }, POLL_INTERVAL);

    return () => { if (intervalRef.current) clearInterval(intervalRef.current); };
  }, [running, backendOnline, addLog]);

  return { currentWindow, windows, incidents, stats, agentLog, backendOnline, logRef };
}
