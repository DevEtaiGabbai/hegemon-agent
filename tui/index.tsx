import { createCliRenderer, type PasteEvent, type ScrollBoxRenderable } from "@opentui/core";
import { createRoot, useKeyboard, useRenderer, useTerminalDimensions } from "@opentui/react";
import { useCallback, useEffect, useRef, useState } from "react";
import { spawn, type ChildProcessWithoutNullStreams } from "node:child_process";
import { resolve } from "node:path";

// The bridge is deliberately the only integration point with the Python agent.
const root = resolve(import.meta.dir, "..");
const python = process.argv[2] || process.env.HEGEMON_PYTHON || "python3";
const initialArgs = process.argv.slice(3);

type Prompt = { prompt: string; default?: string; secret?: boolean };
type Field = "search" | "batch-topic" | "batch-key" | "build";
type Status = "ready" | "running" | "waiting" | "success" | "failed" | "cancelled";
const C = { bg: "#0b1210", panel: "#101c18", panel2: "#14231d", line: "#294137", mint: "#9ce6c2", green: "#54c990", white: "#e1eee5", muted: "#87a396", dim: "#587266", amber: "#eacb8b", red: "#f19c91" };
const actions = [
  ["0", "Agent", "Build a page from a topic"],
  ["1", "Search", "Find and select studies"],
  ["2", "Batch", "Build a shared topic page"],
  ["3", "Build", "Start from an accession"],
  ["4", "Diagnose", "Inspect site"],
  ["5", "Check", "Verify setup"],
  ["6", "Mode", "Choose AI mode"],
  ["7", "Key", "Set API key"],
] as const;

function App() {
  const renderer = useRenderer();
  const { width, height } = useTerminalDimensions();
  const compact = width < 78;
  const tiny = height < 25;
  const dense = compact || tiny;
  const [view, setView] = useState<"home" | "field" | "run">("home");
  const [field, setField] = useState<Field>("search");
  const [input, setInput] = useState("");
  const [batchTopic, setBatchTopic] = useState("");
  const [prompt, setPrompt] = useState<Prompt | null>(null);
  const [status, setStatus] = useState<Status>("ready");
  const [active, setActive] = useState(false);
  const [command, setCommand] = useState<string[]>([]);
  const [log, setLog] = useState("");
  const [notice, setNotice] = useState("");
  const proc = useRef<ChildProcessWithoutNullStreams | null>(null);
  const scroll = useRef<ScrollBoxRenderable>(null);
  const cancelled = useRef(false);
  const started = useRef(false);

  const append = useCallback((text: string) => {
    // Terminal escape sequences are not part of the JSONL protocol and must not
    // be allowed to control the renderer. Keep partial chunks in their order.
    const safe = text.replace(/\x1b\[[0-?]*[ -/]*[@-~]/g, "").replace(/\r/g, "\n").replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, "");
    setLog(previous => (previous + safe).slice(-200000));
  }, []);

  const cancel = useCallback(() => {
    const child = proc.current;
    if (!child) return;
    cancelled.current = true;
    setPrompt(null);
    setInput("");
    setStatus("cancelled");
    append("\n\n  Interrupted. Partial files may remain in the build folder.\n");
    // A separate process group also stops any subprocesses started by Python.
    try { if (child.pid) process.kill(-child.pid, "SIGTERM"); else child.kill("SIGTERM"); }
    catch { child.kill("SIGTERM"); }
    const timer = setTimeout(() => {
      if (proc.current === child) {
        try { if (child.pid) process.kill(-child.pid, "SIGKILL"); else child.kill("SIGKILL"); }
        catch { /* already exited */ }
      }
    }, 1500);
    timer.unref();
  }, [append]);

  const run = useCallback((args: string[]) => {
    if (proc.current) return;
    setView("run"); setCommand(args); setLog(""); setPrompt(null); setInput(""); setNotice(""); setStatus("running");
    cancelled.current = false;
    let child: ChildProcessWithoutNullStreams;
    try {
      child = spawn(python, [resolve(root, "tui/bridge.py"), ...args], { cwd: root, stdio: "pipe", detached: true });
    } catch (error) {
      append(`Unable to start bridge: ${String(error)}\n`); setStatus("failed"); return;
    }
    proc.current = child;
    setActive(true);
    let pending = "";
    let exitReceived = false;
    const handleLine = (line: string) => {
      if (!line.trim()) return;
      try {
        const message = JSON.parse(line) as { type?: string; text?: string; prompt?: string; default?: string; secret?: boolean; code?: number };
        if (message.type === "output" && typeof message.text === "string") append(message.text);
        else if (message.type === "prompt" && typeof message.prompt === "string" && !cancelled.current) {
          setPrompt({ prompt: message.prompt, default: message.default, secret: message.secret }); setInput(""); setStatus("waiting");
        } else if (message.type === "exit" && typeof message.code === "number") {
          exitReceived = true;
          setPrompt(null);
          if (!cancelled.current) setStatus(message.code === 0 ? "success" : "failed");
          append(`\n\n  Process exited with code ${message.code}.\n`);
        } else append(`\n[bridge] Unexpected message: ${line}\n`);
      } catch { append(`\n[bridge] Invalid JSON: ${line}\n`); }
    };
    child.stdout.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      pending += chunk;
      const lines = pending.split("\n"); pending = lines.pop() || "";
      lines.forEach(handleLine);
    });
    child.stderr.setEncoding("utf8");
    child.stderr.on("data", (chunk: string) => append(chunk));
    child.on("error", error => { append(`\n[bridge] ${error.message}\n`); if (!cancelled.current) setStatus("failed"); });
    child.on("close", code => {
      if (pending) handleLine(pending);
      if (!exitReceived && !cancelled.current) { append(`\n[bridge] Closed without an exit message (code ${code ?? "unknown"}).\n`); setStatus("failed"); }
      setPrompt(null);
      if (proc.current === child) { proc.current = null; setActive(false); }
    });
    child.stdin.on("error", error => { if (!cancelled.current) append(`\n[bridge stdin] ${error.message}\n`); });
  }, [append]);

  useEffect(() => {
    if (!started.current && initialArgs.length) { started.current = true; run(initialArgs); }
    return () => {
      const child = proc.current;
      if (child) { try { if (child.pid) process.kill(-child.pid, "SIGTERM"); else child.kill(); } catch { /* already exited */ } }
    };
  }, [run]);

  const editing = view === "field" || (view === "run" && !!prompt);
  // Bracketed paste does not arrive as individual key events in OpenTUI.
  useEffect(() => {
    if (!editing) return;
    const onPaste = (event: PasteEvent) => {
      const value = new TextDecoder().decode(event.bytes).replace(/[\r\n]+/g, " ").replace(/[\x00-\x1f\x7f]/g, "");
      setInput(current => current + value); event.preventDefault();
    };
    renderer.keyInput.on("paste", onPaste);
    return () => { renderer.keyInput.off("paste", onPaste); };
  }, [editing, renderer]);

  const beginField = (next: Field) => { setField(next); setInput(""); setNotice(""); setView("field"); };
  const submit = () => {
    if (view === "run" && prompt) {
      const answer = input;
      try { proc.current?.stdin.write(JSON.stringify({ type: "answer", value: answer }) + "\n"); }
      catch (error) { append(`\n[bridge stdin] ${String(error)}\n`); }
      if (!prompt.secret) append(`${prompt.prompt} ${answer || prompt.default || ""}\n`);
      else append(`${prompt.prompt} [hidden]\n`);
      setInput(""); setPrompt(null); setStatus("running"); return;
    }
    if (view !== "field") return;
    const value = input.trim();
    if (field !== "batch-topic" && !value) { setNotice("Please enter a value to continue."); return; }
    if (field === "search") run(["search", value]);
    else if (field === "build") run(["build", value]);
    else if (field === "batch-topic") { setBatchTopic(value); beginField("batch-key"); }
    else run(["batch", ...(batchTopic ? [batchTopic] : []), "--key", value]);
  };

  useKeyboard(key => {
    if (key.ctrl && key.name === "c") {
      if (proc.current) cancel(); else renderer.destroy();
      return;
    }
    if (key.name === "pageup") { scroll.current?.scrollBy(-Math.max(3, height - 12)); return; }
    if (key.name === "pagedown") { scroll.current?.scrollBy(Math.max(3, height - 12)); return; }
    if (key.name === "escape") {
      if (proc.current) cancel();
      else if (view !== "home") { setView("home"); setInput(""); setNotice(""); }
      else renderer.destroy();
      return;
    }
    if (editing) {
      if (key.name === "return") { submit(); return; }
      if (key.name === "backspace" || key.name === "delete") { setInput(value => Array.from(value).slice(0, -1).join("")); return; }
      if (!key.ctrl && !key.meta && !key.option && key.sequence && !/[\x00-\x1f\x7f\x1b]/.test(key.sequence)) setInput(value => value + key.sequence);
      return;
    }
    if (view === "run") {
      if (!proc.current && (key.name === "return" || key.name === "h")) setView("home");
      return;
    }
    if (key.name === "0" || key.name === "a") run(["agent"]);
    else if (key.name === "1" || key.name === "s") beginField("search");
    else if (key.name === "2" || key.name === "b") beginField("batch-topic");
    else if (key.name === "3") beginField("build");
    else if (key.name === "4" || key.name === "d") run(["diagnose"]);
    else if (key.name === "5" || key.name === "c") run(["check"]);
    else if (key.name === "6" || key.name === "m") run(["mode"]);
    else if (key.name === "7" || key.name === "k") run(["key"]);
    else if (key.name === "q") renderer.destroy();
  });

  const fieldTitle = field === "search" ? "Search by topic" : field === "batch-topic" ? "Start a batch" : field === "batch-key" ? "Choose a shared key" : "Build an accession";
  const fieldHint = field === "search" ? "Describe the biology or disease you want to explore." : field === "batch-topic" ? "Enter a topic, or leave blank to use an existing key." : field === "batch-key" ? "Datasets in this batch will share this Hegemon key." : "Enter a GEO or ArrayExpress accession, such as GSE51984.";
  const stateLabel = { ready: "READY", running: "RUNNING", waiting: "INPUT NEEDED", success: "COMPLETE", failed: "FAILED", cancelled: "STOPPED" }[status];
  const stateColor = status === "failed" ? C.red : status === "waiting" ? C.amber : C.mint;
  const displayInput = prompt?.secret && view === "run" ? "•".repeat(Array.from(input).length) : input;

  return (
    <box flexDirection="column" width="100%" height="100%" backgroundColor={C.bg} paddingLeft={compact ? 1 : 2} paddingRight={compact ? 1 : 2}>
      <box height={3} flexDirection="row" alignItems="center" justifyContent="space-between" border={['bottom']} borderColor={C.line}>
        <text fg={C.mint}><strong>◈  HEGEMON</strong><span fg={C.dim}>  /  RESEARCH WORKSTATION</span></text>
        {!compact && <text fg={view === "run" ? stateColor : C.muted}>● {view === "run" ? stateLabel : "WORKSPACE READY"}</text>}
      </box>
      {view === "home" ? (
        <box flexGrow={1} flexDirection="column" paddingTop={dense ? 0 : 1}>
          <text fg={C.green}>RESEARCH  /  HOME</text>
          <box height={dense ? 0 : 2} />
          <text fg={C.white}><strong>From a question to a dataset.</strong></text>
          {!dense && <text fg={C.muted}>Search, build and publish with the Hegemon agent.</text>}
          <box height={dense ? 0 : 1} />
          <box flexDirection={compact ? "column" : "row"} flexGrow={1} gap={dense ? 0 : 1}>
            <box flexDirection="column" flexGrow={compact ? 0 : 1} flexShrink={compact ? 0 : 1} border borderColor={C.line} backgroundColor={C.panel} paddingLeft={1} paddingRight={1}>
              <text fg={C.green}>START A WORKFLOW</text>
              {!dense && <text fg={C.dim}>Choose an action to get started</text>}
              {!dense && <box height={1} />}
              {actions.slice(0, 4).map(([key, name, description]) => <text key={key} fg={C.white}><span fg={C.mint}> {key}  </span><strong>{name.padEnd(9)}</strong><span fg={C.muted}>{compact && width < 48 ? "" : description}</span></text>)}
              {!dense && <box height={1} />}
              {!dense && <text fg={C.dim}>Cloud default: GPT-6.1 Sol · configurable in settings</text>}
            </box>
            <box flexDirection="column" flexShrink={compact ? 0 : 1} width={compact ? "100%" : 34} border borderColor={C.line} backgroundColor={C.panel} paddingLeft={1} paddingRight={1}>
              <text fg={C.green}>WORKSPACE TOOLS</text>
              {!dense && <box height={1} />}
              {actions.slice(4).map(([key, name, description]) => <text key={key} fg={C.white}><span fg={C.mint}> {key}  </span><strong>{name.padEnd(10)}</strong><span fg={C.muted}>{compact && width < 48 ? "" : description}</span></text>)}
            </box>
          </box>
          {!dense && <box height={1} />}
        </box>
      ) : view === "field" ? (
        <box flexGrow={1} flexDirection="column" paddingTop={1}>
          <text fg={C.green}>NEW WORKFLOW  /  {field === "batch-key" ? "STEP 2 OF 2" : field === "batch-topic" ? "STEP 1 OF 2" : "START"}</text>
          <box height={1} />
          <text fg={C.white}><strong>{fieldTitle}</strong></text>
          <text fg={C.muted}>{fieldHint}</text>
          <box height={1} />
          <box border borderColor={C.green} backgroundColor={C.panel2} paddingLeft={1} paddingRight={1} height={3}>
            <text fg={C.white}><span fg={C.mint}>›  </span>{displayInput}<span fg={C.green}>▏</span></text>
          </box>
          <text fg={C.amber}>{notice}</text>
          <box flexGrow={1} />
          <text fg={C.dim}>The Python agent will guide you through the next steps.</text>
          <box height={1} />
        </box>
      ) : (
        <box flexGrow={1} flexDirection="column" paddingTop={tiny ? 0 : 1} minHeight={0}>
          <box flexDirection="row" justifyContent="space-between" height={1}>
            <text fg={C.green}>SESSION  /  {command[0]?.toUpperCase() || "AGENT"}</text>
            <text fg={stateColor}>● {stateLabel}</text>
          </box>
          {!tiny && <text fg={C.muted}>hegemon {command.join(" ")}</text>}
          {!tiny && <box height={1} />}
          <box flexDirection="column" flexGrow={1} minHeight={3} border borderColor={C.line} backgroundColor={C.panel} paddingLeft={1} paddingRight={1}>
            <text fg={C.dim}>ACTIVITY  {compact ? "" : "·  LIVE AGENT OUTPUT"}</text>
            <scrollbox ref={scroll} flexGrow={1} minHeight={1} stickyScroll stickyStart="bottom" scrollX={false}>
              <text fg={C.white}>{log || "Connecting to the Python agent…"}</text>
            </scrollbox>
          </box>
          {prompt ? <box flexDirection="column" flexShrink={0} height={Math.min(Math.max(4, 3 + Math.ceil((prompt.prompt.length + (prompt.default?.length || 0)) / Math.max(10, width - 8))), Math.max(4, Math.floor(height / 2)))} border borderColor={C.green} backgroundColor={C.panel2} paddingLeft={1} paddingRight={1}>
            <text fg={C.mint}>{prompt.prompt}{prompt.default && !prompt.secret ? ` [${prompt.default}]` : ""}</text>
            <text fg={C.white}><span fg={C.green}>›  </span>{displayInput}<span fg={C.green}>▏</span>{prompt.secret ? <span fg={C.dim}>  hidden input</span> : null}</text>
          </box> : <text fg={C.muted}>{active ? "Agent is working · output will appear above" : "Session ended · press Enter to return home"}</text>}
        </box>
      )}
      <box height={2} flexDirection="column" justifyContent="center" border={['top']} borderColor={C.line}>
        <text fg={C.dim}>{view === "home" ? "0–7 actions   ·   q quit" : view === "field" ? "Enter continue   ·   Esc home" : compact ? "PgUp/PgDn scroll   ·   Esc stop / home" : "PgUp/PgDn scroll   ·   Esc stop / home   ·   Ctrl+C stop / quit"}</text>
      </box>
    </box>
  );
}

const renderer = await createCliRenderer({ exitOnCtrlC: false });
createRoot(renderer).render(<App />);
