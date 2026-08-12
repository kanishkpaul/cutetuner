import {
  Activity, ArrowDownToLine, AudioLines, Check, ChevronRight, CircleAlert, Disc3,
  FileAudio, Gauge, Headphones, Library, LoaderCircle, LockKeyhole, Mic2, Pause,
  Play, Plus, RotateCcw, Settings2, SlidersHorizontal, Sparkles, Trash2, WandSparkles,
  X,
} from "lucide-react";
import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import WaveSurfer from "wavesurfer.js";
import { api } from "./api";
import type {
  AnalysisReport, CreativeBrief, InputMode, Job, Project, SystemInfo, TuningControls,
} from "./types";

const modeLabels: Record<InputMode, string> = {
  full_mix: "Full song",
  vocal_backing: "Vocal + backing",
  vocal_only: "Dry vocal",
};

function classNames(...values: Array<string | false | undefined>) {
  return values.filter(Boolean).join(" ");
}

function formatTime(seconds: number) {
  const minutes = Math.floor(seconds / 60);
  return `${minutes}:${Math.round(seconds % 60).toString().padStart(2, "0")}`;
}

function Confidence({ value }: { value: number }) {
  const label = value >= 0.7 ? "high" : value >= 0.4 ? "medium" : "low";
  return <span className={`confidence ${label}`}>{label} confidence</span>;
}

function Brand() {
  return (
    <div className="brand">
      <div className="brand-mark"><AudioLines size={19} strokeWidth={1.8} /></div>
      <div><strong>CUTE</strong><span>Tuner</span></div>
    </div>
  );
}

function ProjectRail({ projects, selected, onSelect, onNew }: {
  projects: Project[]; selected?: string; onSelect: (id: string) => void; onNew: () => void;
}) {
  return (
    <aside className="rail">
      <Brand />
      <button className="new-project" onClick={onNew}><Plus size={17} /> New session</button>
      <div className="rail-label">Recent sessions</div>
      <nav className="project-list" aria-label="Recent sessions">
        {projects.map((project) => (
          <button key={project.id} className={classNames("project-row", selected === project.id && "active")} onClick={() => onSelect(project.id)}>
            <span className="project-icon"><FileAudio size={16} /></span>
            <span className="project-copy"><strong>{project.name}</strong><small>{modeLabels[project.mode]} · {project.status}</small></span>
            <ChevronRight size={14} />
          </button>
        ))}
        {!projects.length && <p className="empty-rail">Your private sessions will appear here.</p>}
      </nav>
      <div className="local-badge"><LockKeyhole size={14} /><span><strong>Local only</strong><small>No cloud uploads</small></span></div>
    </aside>
  );
}

function DropField({ label, hint, file, onFile }: { label: string; hint: string; file: File | null; onFile: (file: File) => void }) {
  const input = useRef<HTMLInputElement>(null);
  const [dragging, setDragging] = useState(false);
  return (
    <button
      type="button"
      className={classNames("drop-field", dragging && "dragging", Boolean(file) && "filled")}
      onClick={() => input.current?.click()}
      onDragOver={(event) => { event.preventDefault(); setDragging(true); }}
      onDragLeave={() => setDragging(false)}
      onDrop={(event) => {
        event.preventDefault(); setDragging(false);
        const next = event.dataTransfer.files[0]; if (next) onFile(next);
      }}
    >
      <input ref={input} type="file" accept="audio/*,.wav,.flac,.mp3,.m4a,.aac" hidden onChange={(event) => event.target.files?.[0] && onFile(event.target.files[0])} />
      {file ? <><Disc3 size={23} /><span><strong>{file.name}</strong><small>{(file.size / 1024 / 1024).toFixed(1)} MB · click to replace</small></span><Check size={18} /></> :
        <><FileAudio size={23} /><span><strong>{label}</strong><small>{hint}</small></span><Plus size={18} /></>}
    </button>
  );
}

function NewProject({ onCreate, busy }: { onCreate: (form: FormData) => Promise<void>; busy: boolean }) {
  const [mode, setMode] = useState<InputMode>("full_mix");
  const [name, setName] = useState("");
  const [primary, setPrimary] = useState<File | null>(null);
  const [backing, setBacking] = useState<File | null>(null);
  const submit = async () => {
    if (!primary || (mode === "vocal_backing" && !backing)) return;
    const form = new FormData();
    form.append("name", name.trim() || primary.name.replace(/\.[^.]+$/, ""));
    form.append("mode", mode); form.append("primary", primary);
    if (backing) form.append("backing", backing);
    await onCreate(form);
  };
  return (
    <main className="welcome-shell">
      <div className="welcome-copy">
        <span className="eyebrow"><Sparkles size={14} /> private vocal intelligence</span>
        <h1>Make the take feel<br /><em>finished.</em></h1>
        <p>CUTE listens for musical intent, asks what you want the song to feel like, then tunes and polishes without sanding away your voice.</p>
        <div className="trust-row"><span>All processing stays here</span><span>Advanced control stays yours</span></div>
      </div>
      <section className="upload-card">
        <div className="step-kicker">Start a session</div>
        <h2>What are we listening to?</h2>
        <div className="mode-grid" role="radiogroup" aria-label="Input type">
          {([
            ["full_mix", "Full song", "One mixed file"],
            ["vocal_backing", "Vocal + backing", "Cleanest result"],
            ["vocal_only", "Dry vocal", "Key required later"],
          ] as const).map(([value, label, hint]) => (
            <button key={value} role="radio" aria-checked={mode === value} className={classNames("mode-option", mode === value && "selected")} onClick={() => { setMode(value); setPrimary(null); setBacking(null); }}>
              <span className="radio-dot" /><strong>{label}</strong><small>{hint}</small>
            </button>
          ))}
        </div>
        <label className="field-label">Session name <span>optional</span></label>
        <input className="text-input" value={name} onChange={(event) => setName(event.target.value)} placeholder="Late-night chorus" maxLength={120} />
        <DropField
          label={mode === "full_mix" ? "Drop the full song" : "Drop the dry vocal"}
          hint="WAV, FLAC, MP3, M4A or AAC · up to 500 MB"
          file={primary} onFile={setPrimary}
        />
        {mode === "vocal_backing" && <DropField label="Drop the backing track" hint="Exported from the same start point" file={backing} onFile={setBacking} />}
        <button className="primary-action" disabled={busy || !primary || (mode === "vocal_backing" && !backing)} onClick={submit}>
          {busy ? <LoaderCircle className="spin" size={18} /> : <Headphones size={18} />}
          {busy ? "Preparing session…" : "Listen to this take"}
        </button>
        <p className="privacy-note"><LockKeyhole size={13} /> Audio is copied only into CUTE Tuner’s private local project folder.</p>
      </section>
    </main>
  );
}

function JobPanel({ job, onCancel }: { job: Job; onCancel: () => void }) {
  return (
    <section className={classNames("job-panel", job.status === "failed" && "failed")} aria-live="polite">
      <div className="job-orbit"><div className="orbit outer" /><div className="orbit inner" /><AudioLines size={28} /></div>
      <div className="job-copy">
        <span className="eyebrow">{job.kind === "analysis" ? "Listening locally" : job.kind === "preview" ? "Making comparisons" : "Finishing your vocal"}</span>
        <h2>{job.status === "failed" ? "This pass needs attention" : job.message}</h2>
        {job.error && <p className="error-copy">{job.error}</p>}
        <div className="progress-track"><div style={{ width: `${job.progress * 100}%` }} /></div>
        <div className="progress-meta"><span>{job.stage.replaceAll("_", " ")}</span><span>{Math.round(job.progress * 100)}%</span></div>
      </div>
      {job.status === "running" && <button className="quiet-button" onClick={onCancel}><X size={15} /> Cancel</button>}
    </section>
  );
}

function MetricCard({ label, metric }: { label: string; metric: { value: string | number | null; unit: string; confidence: number; source: string } }) {
  return (
    <div className="metric-card">
      <span>{label.replaceAll("_", " ")}</span>
      <strong>{metric.value ?? "—"}<small>{metric.unit}</small></strong>
      <div className="metric-foot"><Confidence value={metric.confidence} /><span>{metric.source}</span></div>
    </div>
  );
}

function PitchCanvas({ report }: { report: AnalysisReport }) {
  const canvas = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const element = canvas.current; if (!element) return;
    const context = element.getContext("2d"); if (!context) return;
    const ratio = window.devicePixelRatio || 1;
    const width = element.clientWidth; const height = element.clientHeight;
    element.width = width * ratio; element.height = height * ratio; context.scale(ratio, ratio);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = "rgba(255,255,255,.06)"; context.lineWidth = 1;
    for (let y = 20; y < height; y += 24) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
    if (!report.notes.length) return;
    const pitches = report.notes.map((note) => note.midi);
    const low = Math.min(...pitches) - 1; const high = Math.max(...pitches) + 1;
    context.lineCap = "round"; context.lineWidth = 5;
    report.notes.forEach((note) => {
      const x1 = (note.start / report.duration_seconds) * width;
      const x2 = (note.end / report.duration_seconds) * width;
      const y = height - 18 - ((note.midi - low) / Math.max(1, high - low)) * (height - 36);
      const gradient = context.createLinearGradient(x1, 0, x2, 0);
      gradient.addColorStop(0, "#a6e8c5"); gradient.addColorStop(1, "#d9f36a");
      context.strokeStyle = gradient; context.beginPath(); context.moveTo(x1, y); context.lineTo(Math.max(x1 + 3, x2), y); context.stroke();
    });
  }, [report]);
  return <canvas ref={canvas} className="pitch-canvas" aria-label="Detected vocal note timeline" />;
}

function BriefForm({ report, onSubmit, busy }: { report: AnalysisReport; onSubmit: (brief: CreativeBrief) => Promise<void>; busy: boolean }) {
  const [vibe, setVibe] = useState("");
  const [delivery, setDelivery] = useState("honest, close and controlled");
  const [lyrics, setLyrics] = useState("");
  const [reference, setReference] = useState("");
  const [protectStart, setProtectStart] = useState("");
  const [protectEnd, setProtectEnd] = useState("");
  const [intent, setIntent] = useState<CreativeBrief["tuning_intent"]>("transparent");
  return (
    <section className="interview-card">
      <div className="section-heading"><span className="step-number">02</span><div><span className="eyebrow">Producer brief</span><h2>What should this performance feel like?</h2></div></div>
      <label className="field-label">Describe the vibe <span>required</span></label>
      <textarea className="large-input" value={vibe} onChange={(event) => setVibe(event.target.value)} placeholder="Intimate, aching, quiet confidence — like the room disappears around the voice…" />
      <div className="two-column-fields">
        <div><label className="field-label">Delivery</label><input className="text-input" value={delivery} onChange={(event) => setDelivery(event.target.value)} /></div>
        <div><label className="field-label">Reference in words <span>optional</span></label><input className="text-input" value={reference} onChange={(event) => setReference(event.target.value)} placeholder="Warm piano ballad, vocal very close" /></div>
      </div>
      <label className="field-label">Lyrics <span>optional · kept local</span></label>
      <textarea className="lyrics-input" value={lyrics} onChange={(event) => setLyrics(event.target.value)} placeholder="Paste lyrics if their emotional arc should influence the plan." />
      <label className="field-label">Protect a passage <span>optional · seconds</span></label>
      <div className="two-column-fields"><input className="text-input" type="number" min="0" step="0.1" value={protectStart} onChange={(event) => setProtectStart(event.target.value)} placeholder="Start, e.g. 42.5" /><input className="text-input" type="number" min="0" step="0.1" value={protectEnd} onChange={(event) => setProtectEnd(event.target.value)} placeholder="End, e.g. 47.0" /></div>
      <label className="field-label">How audible should tuning be?</label>
      <div className="intent-grid">
        {(["transparent", "balanced", "obvious"] as const).map((value) => <button key={value} className={classNames("intent-option", intent === value && "selected")} onClick={() => setIntent(value)}><strong>{value}</strong><small>{value === "transparent" ? "Preserve every human edge" : value === "balanced" ? "Clean, still believable" : "A deliberate effect"}</small></button>)}
      </div>
      <button className="primary-action" disabled={busy || vibe.trim().length < 2 || Boolean(protectStart && protectEnd && Number(protectEnd) <= Number(protectStart))} onClick={() => onSubmit({ vibe, delivery, lyrics: lyrics || null, reference_description: reference || null, tuning_intent: intent, protected_ranges: protectStart && protectEnd ? [{ start: Number(protectStart), end: Number(protectEnd) }] : [] })}>
        <WandSparkles size={18} /> Build my tuning plan
      </button>
      {!report.key && <p className="inline-warning"><CircleAlert size={15} /> With no backing track, the studio will ask you to choose a key before major/minor correction.</p>}
    </section>
  );
}

function ListeningReport({ project, onBrief, busy }: { project: Project; onBrief: (brief: CreativeBrief) => Promise<void>; busy: boolean }) {
  const report = project.report!;
  return (
    <div className="content-column">
      <section className="report-hero">
        <div className="section-heading"><span className="step-number">01</span><div><span className="eyebrow">What I hear</span><h1>{project.name}</h1></div></div>
        <div className="descriptor-row">{report.semantic_descriptors.map((item) => <span key={item}>{item}</span>)}</div>
        <div className="heard-grid">
          <div className="heard-copy">{report.what_i_hear.map((line) => <p key={line}>{line}</p>)}</div>
          <div className="key-orb"><small>Harmonic centre</small><strong>{report.key ? `${report.key} ${report.scale}` : "Needs context"}</strong><Confidence value={report.key_confidence} /></div>
        </div>
        {report.warnings.map((warning) => <p className="inline-warning" key={warning}><CircleAlert size={15} /> {warning}</p>)}
        <PitchCanvas report={report} />
        <div className="timeline-meta"><span>{report.notes.length} note events</span><span>{formatTime(report.duration_seconds)}</span><span>{report.chords.length} harmonic regions</span></div>
      </section>
      <section className="metrics-section">
        <div className="section-title"><Activity size={17} /><h3>Evidence, not a verdict</h3><span>Every estimate shows its source and confidence.</span></div>
        <div className="metric-grid">
          {Object.entries({ ...report.vocal, naturalness: report.naturalness, ...report.technical }).slice(0, 8).map(([label, metric]) => <MetricCard key={label} label={label} metric={metric} />)}
        </div>
      </section>
      <BriefForm report={report} onSubmit={onBrief} busy={busy} />
    </div>
  );
}

function RangeControl({ label, hint, value, min, max, step, suffix = "", onChange }: {
  label: string; hint: string; value: number; min: number; max: number; step: number; suffix?: string; onChange: (value: number) => void;
}) {
  const percent = ((value - min) / (max - min)) * 100;
  return (
    <label className="range-control">
      <span><strong>{label}</strong><small>{hint}</small></span>
      <div className="range-line"><input type="range" min={min} max={max} step={step} value={value} onChange={(event) => onChange(Number(event.target.value))} style={{ "--range": `${percent}%` } as React.CSSProperties} /><output>{Number.isInteger(step) ? Math.round(value) : value.toFixed(2)}{suffix}</output></div>
    </label>
  );
}

function CompareDeck({ project }: { project: Project }) {
  const variants = useMemo(() => {
    const values = [{ key: "original", label: "Original", url: `/api/projects/${project.id}/audio/vocal` }];
    project.outputs.filter((item) => item.kind.startsWith("preview_")).forEach((item) => values.push({ key: item.kind, label: item.kind.replace("preview_", ""), url: `/api/projects/${project.id}/outputs/${item.kind}` }));
    project.outputs.filter((item) => !item.kind.startsWith("preview_")).forEach((item) => values.push({ key: item.kind, label: item.kind === "finished_mix" ? "Finished mix" : "Corrected vocal", url: `/api/projects/${project.id}/outputs/${item.kind}` }));
    return values;
  }, [project]);
  const [selected, setSelected] = useState(variants[0]?.key);
  const [playing, setPlaying] = useState(false);
  const waveContainer = useRef<HTMLDivElement>(null);
  const wave = useRef<WaveSurfer | null>(null);
  const rememberedTime = useRef(0);
  const selectedVariant = variants.find((item) => item.key === selected) ?? variants[0];
  useEffect(() => {
    if (!waveContainer.current || !selectedVariant) return;
    const instance = WaveSurfer.create({ container: waveContainer.current, waveColor: "#5f6963", progressColor: "#d9f36a", cursorColor: "#f5f6ed", height: 76, barWidth: 2, barGap: 2, barRadius: 2, normalize: true });
    wave.current = instance; void instance.load(selectedVariant.url).catch((error: Error) => { if (error.name !== "AbortError") console.error(error); });
    instance.on("ready", () => { if (rememberedTime.current) instance.setTime(rememberedTime.current); if (playing) instance.play(); });
    instance.on("timeupdate", (time) => { rememberedTime.current = time; });
    instance.on("finish", () => setPlaying(false));
    return () => { rememberedTime.current = instance.getCurrentTime(); instance.destroy(); };
  }, [selectedVariant?.url]);
  return (
    <div className="compare-deck">
      <div className="compare-tabs">{variants.map((item) => <button key={item.key} className={classNames(selected === item.key && "active")} onClick={() => setSelected(item.key)}>{item.label}</button>)}</div>
      <div className="wave-row"><button className="round-play" onClick={() => { wave.current?.playPause(); setPlaying(!playing); }} aria-label={playing ? "Pause preview" : "Play preview"}>{playing ? <Pause size={18} /> : <Play size={18} fill="currentColor" />}</button><div className="waveform" ref={waveContainer} /></div>
      <p>Switch versions while listening; playback stays at the same moment for an honest comparison.</p>
    </div>
  );
}

function Studio({ project, onSave, onPreviews, onRender, busy }: {
  project: Project; onSave: (controls: TuningControls) => Promise<void>; onPreviews: () => Promise<void>; onRender: (controls: TuningControls) => Promise<void>; busy: boolean;
}) {
  const [controls, setControls] = useState<TuningControls>(project.plan!.controls);
  useEffect(() => setControls(project.plan!.controls), [project.plan]);
  const set = (key: keyof TuningControls, value: number | string | null) => setControls((current) => ({ ...current, [key]: value }));
  const previews = project.outputs.some((output) => output.kind.startsWith("preview_"));
  return (
    <div className="studio-layout">
      <section className="studio-main">
        <div className="section-heading"><span className="step-number">03</span><div><span className="eyebrow">Tuning studio</span><h1>Shape the correction</h1></div></div>
        <div className="plan-rationale">{project.plan!.rationale.map((line) => <p key={line}><Sparkles size={14} />{line}</p>)}</div>
        <PitchCanvas report={project.report!} />
        {previews ? <CompareDeck project={project} /> : <div className="preview-empty"><Headphones size={26} /><div><strong>Hear three honest versions</strong><p>Natural, recommended, and tighter previews use the same passage and loudness.</p></div><button className="secondary-action" disabled={busy} onClick={async () => { await onSave(controls); await onPreviews(); }}><Play size={16} /> Render previews</button></div>}
        <div className="render-banner"><div><span className="eyebrow">Ready when you are</span><h3>{project.mode === "vocal_only" ? "Export the corrected vocal." : "Export the corrected vocal and finished mix."}</h3><p>Masters stay local. Downloads are 320 kbps MP3.</p></div><button className="primary-action compact" disabled={busy || (!controls.key && controls.scale !== "chromatic")} onClick={() => onRender(controls)}><WandSparkles size={18} /> Render final song</button></div>
      </section>
      <aside className="control-panel">
        <div className="control-heading"><SlidersHorizontal size={18} /><div><strong>Advanced control</strong><small>Every automatic choice remains editable</small></div></div>
        <div className="select-row"><label>Key<select value={controls.key ?? ""} onChange={(event) => set("key", event.target.value || null)}><option value="">Choose</option>{["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"].map((key) => <option key={key}>{key}</option>)}</select></label><label>Scale<select value={controls.scale ?? "chromatic"} onChange={(event) => set("scale", event.target.value)}><option value="major">Major</option><option value="minor">Minor</option><option value="chromatic">Chromatic</option></select></label></div>
        <RangeControl label="Correction" hint="How strongly notes move" value={controls.strength} min={0} max={1} step={0.01} onChange={(value) => set("strength", value)} />
        <RangeControl label="Retune speed" hint="Slow stays human" value={controls.retune_ms} min={5} max={300} step={1} suffix=" ms" onChange={(value) => set("retune_ms", value)} />
        <RangeControl label="Maximum movement" hint="Caps surprising corrections" value={controls.max_correction_cents} min={10} max={600} step={5} suffix="¢" onChange={(value) => set("max_correction_cents", value)} />
        <div className="control-divider" /><h4>Human detail</h4>
        <RangeControl label="Vibrato preservation" hint="Keep note movement" value={controls.vibrato_preservation} min={0} max={1} step={0.01} onChange={(value) => set("vibrato_preservation", value)} />
        <RangeControl label="Formant shift" hint="Zero keeps your tone" value={controls.formant_shift} min={-2} max={2} step={0.1} onChange={(value) => set("formant_shift", value)} />
        <RangeControl label="Breath protection" hint="Avoid metallic breaths" value={controls.breath_protection} min={0} max={1} step={0.01} onChange={(value) => set("breath_protection", value)} />
        <div className="control-divider" /><h4>Vocal finish</h4>
        <RangeControl label="EQ" hint="Clean low rumble" value={controls.eq_amount} min={0} max={1} step={0.01} onChange={(value) => set("eq_amount", value)} />
        <RangeControl label="De-esser" hint="Soften sharp consonants" value={controls.deesser_amount} min={0} max={1} step={0.01} onChange={(value) => set("deesser_amount", value)} />
        <RangeControl label="Compression" hint="Settle level changes" value={controls.compression_amount} min={0} max={1} step={0.01} onChange={(value) => set("compression_amount", value)} />
        <RangeControl label="Ambience" hint="Short private room" value={controls.ambience_amount} min={0} max={1} step={0.01} onChange={(value) => set("ambience_amount", value)} />
        <RangeControl label="Vocal level" hint="Level in the final mix" value={controls.vocal_gain_db} min={-12} max={12} step={0.5} suffix=" dB" onChange={(value) => set("vocal_gain_db", value)} />
        <button className="save-settings" onClick={() => onSave(controls)} disabled={busy}><Check size={16} /> Save current plan</button>
      </aside>
    </div>
  );
}

function Results({ project, onRerender, onRate }: { project: Project; onRerender: () => void; onRate: (winner: string) => void }) {
  const finals = project.outputs.filter((item) => !item.kind.startsWith("preview_"));
  return (
    <div className="content-column results-page">
      <section className="result-hero">
        <span className="success-mark"><Check size={28} /></span><span className="eyebrow">Render complete</span>
        <h1>Your vocal still sounds like you.</h1><p>{project.mode === "vocal_only" ? "The vocal master was made locally from the approved plan." : "Both masters were made locally from the approved plan."} Nothing was uploaded.</p>
      </section>
      <CompareDeck project={project} />
      <section className="download-grid">
        {finals.map((output) => <a className="download-card" key={output.kind} href={`/api/projects/${project.id}/outputs/${output.kind}`} download><span className="download-icon"><ArrowDownToLine size={22} /></span><span><small>{output.kind === "finished_mix" ? "Complete song" : "Vocal stem"}</small><strong>{output.filename}</strong><em>MP3 · 320 kbps</em></span><ChevronRight size={18} /></a>)}
      </section>
      <section className="feedback-card"><div><span className="eyebrow">Teach your private tuner</span><h3>Which version serves the song?</h3><p>Your choice stays on this Mac and begins personalizing previews after 20 comparisons.</p></div><div className="feedback-actions">{["original", "natural", "recommended", "tighter"].map((value) => <button key={value} onClick={() => onRate(value)}>{value}</button>)}</div></section>
      <button className="quiet-button centred" onClick={onRerender}><RotateCcw size={15} /> Return to controls and rerender</button>
    </div>
  );
}

function systemStatus(key: string, value: string | number | boolean | null) {
  if (key === "memory_free_percent" && typeof value === "number") return `${value}% free`;
  if (key === "heavy_tasks_ready") return value ? "ready" : "paused for RAM";
  if (typeof value === "string") return "found";
  return value ? "available" : "not configured";
}

function SettingsDrawer({ system, onClose, onDeleteAll, onRefresh }: { system: SystemInfo | null; onClose: () => void; onDeleteAll: () => void; onRefresh: () => Promise<void> }) {
  const [endpoint, setEndpoint] = useState(system?.llm_config.endpoint ?? "");
  const [model, setModel] = useState(system?.llm_config.model ?? "");
  const saveLlm = async () => { await api.configureLlm(endpoint.trim() || null, model.trim() || null); await onRefresh(); };
  return <div className="drawer-backdrop" onMouseDown={onClose}><aside className="settings-drawer" onMouseDown={(event) => event.stopPropagation()}><div className="drawer-title"><div><span className="eyebrow">Local system</span><h2>Models & privacy</h2></div><button onClick={onClose}><X size={19} /></button></div><div className="system-list">{system && Object.entries(system.models).filter(([key]) => key !== "quality_install").map(([key, value]) => <div key={key}><span>{key.replaceAll("_", " ")}</span><strong className={value ? "available" : "missing"}>{systemStatus(key, value)}</strong></div>)}</div><div className="model-note"><Gauge size={18} /><div><strong>5 GB model budget</strong><p>Heavy analyzers are isolated and loaded one at a time. CUTE checks memory before model work and falls back safely.</p></div></div><code>./scripts/setup_quality.sh</code><div className="control-divider" /><label className="field-label">Local producer endpoint</label><input className="text-input" value={endpoint} onChange={(event) => setEndpoint(event.target.value)} placeholder="http://127.0.0.1:8080/v1" /><label className="field-label">Loaded model name</label><input className="text-input" value={model} onChange={(event) => setModel(event.target.value)} placeholder="your-local-gguf" /><button className="save-settings" onClick={saveLlm}><Check size={16} /> Save local model</button><label className="master-toggle"><input type="checkbox" checked={system?.keep_masters ?? true} onChange={async (event) => { await api.keepMasters(event.target.checked); await onRefresh(); }} /><span>Keep 24-bit WAV masters for rerenders</span></label><div className="danger-zone"><strong>All local data</strong><p>Remove every session, output, private rating, downloaded model, and isolated runtime.</p><button onClick={onDeleteAll}><Trash2 size={15} /> Delete all local data</button></div></aside></div>;
}

export default function App() {
  const [projects, setProjects] = useState<Project[]>([]);
  const [project, setProject] = useState<Project | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [system, setSystem] = useState<SystemInfo | null>(null);
  const [settings, setSettings] = useState(false);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");
  const refreshList = useCallback(async () => setProjects(await api.projects()), []);
  const refreshProject = useCallback(async (id: string) => { const fresh = await api.project(id); setProject(fresh); await refreshList(); }, [refreshList]);
  useEffect(() => { Promise.all([api.projects(), api.system()]).then(([nextProjects, nextSystem]) => { setProjects(nextProjects); setSystem(nextSystem); }).catch((reason) => setError(reason.message)); }, []);
  const watch = useCallback((nextJob: Job) => {
    setJob(nextJob);
    const source = new EventSource(`/api/jobs/${nextJob.id}/events`);
    source.onmessage = (event) => {
      const state = JSON.parse(event.data) as Job; setJob(state);
      if (["complete", "failed", "cancelled"].includes(state.status)) {
        source.close(); setBusy(false);
        if (state.status === "complete") refreshProject(state.project_id).then(() => setTimeout(() => setJob(null), 650));
      }
    };
    source.onerror = () => { source.close(); setBusy(false); };
  }, [refreshProject]);
  const create = async (form: FormData) => {
    setBusy(true); setError("");
    try { const created = await api.create(form); setProject(created); await refreshList(); watch(await api.analyze(created.id)); }
    catch (reason) { setBusy(false); setError(reason instanceof Error ? reason.message : "Could not create the session"); }
  };
  const handle = async (work: () => Promise<void>) => { setBusy(true); setError(""); try { await work(); } catch (reason) { setError(reason instanceof Error ? reason.message : "Something went wrong"); } finally { setBusy(false); } };
  const deleteCurrent = async () => { if (!project || !confirm(`Delete “${project.name}” and its local audio?`)) return; await api.remove(project.id); setProject(null); await refreshList(); };
  const content = !project ? <NewProject onCreate={create} busy={busy} /> : job && ["queued", "running", "failed"].includes(job.status) ? <main className="work-shell"><JobPanel job={job} onCancel={() => api.cancel(job.id)} /></main> : project.report && !project.plan ? <ListeningReport project={project} busy={busy} onBrief={(brief) => handle(async () => { await api.brief(project.id, brief); await refreshProject(project.id); })} /> : project.plan && project.status !== "complete" ? <Studio project={project} busy={busy} onSave={(controls) => handle(async () => { await api.plan(project.id, controls); await refreshProject(project.id); })} onPreviews={async () => { setBusy(true); watch(await api.previews(project.id)); }} onRender={async (controls) => { setBusy(true); watch(await api.render(project.id, controls)); }} /> : project.status === "complete" ? <Results project={project} onRerender={() => setProject({ ...project, status: "previewed" })} onRate={(winner) => handle(async () => { await api.rate(project.id, winner); setSystem(await api.system()); })} /> : <main className="work-shell"><button className="primary-action" onClick={async () => { setBusy(true); watch(await api.analyze(project.id)); }}><Headphones size={18} /> Analyze this session</button></main>;
  return (
    <div className="app-shell">
      <ProjectRail projects={projects} selected={project?.id} onSelect={(id) => handle(async () => refreshProject(id))} onNew={() => setProject(null)} />
      <div className="main-frame">
        <header className="topbar"><div className="mobile-brand"><Brand /></div><div className="session-state">{project ? <><span className="status-dot" />{project.name}<small>{project.status}</small></> : <span>Local vocal studio</span>}</div><div className="top-actions">{project && <button onClick={deleteCurrent} title="Delete this session"><Trash2 size={17} /></button>}<button onClick={() => setSettings(true)} title="Settings"><Settings2 size={18} /></button></div></header>
        {error && <div className="global-error"><CircleAlert size={16} /><span>{error}</span><button onClick={() => setError("")}><X size={15} /></button></div>}
        <div className="page-scroll">{content}</div>
      </div>
      {settings && <SettingsDrawer system={system} onClose={() => setSettings(false)} onRefresh={async () => setSystem(await api.system())} onDeleteAll={async () => { if (!confirm("Delete every CUTE Tuner session, output, private rating, and downloaded model?")) return; await api.removeAll(); setProject(null); setSettings(false); await refreshList(); }} />}
    </div>
  );
}
