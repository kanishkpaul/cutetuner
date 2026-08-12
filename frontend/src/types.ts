export type InputMode = "full_mix" | "vocal_backing" | "vocal_only";

export interface Metric {
  value: number | string | null;
  unit: string;
  confidence: number;
  source: string;
}

export interface NoteEvent {
  start: number;
  end: number;
  midi: number;
  target_midi: number | null;
  confidence: number;
}

export interface ChordEvent {
  start: number;
  end: number;
  label: string;
  confidence: number;
}

export interface AnalysisReport {
  duration_seconds: number;
  sample_rate: number;
  channels: number;
  key: string | null;
  scale: string | null;
  key_confidence: number;
  key_alternatives: string[];
  technical: Record<string, Metric>;
  vocal: Record<string, Metric>;
  semantic_descriptors: string[];
  naturalness: Metric;
  notes: NoteEvent[];
  chords: ChordEvent[];
  what_i_hear: string[];
  warnings: string[];
  analyzer_status: Record<string, string>;
}

export interface CreativeBrief {
  vibe: string;
  delivery: string;
  lyrics: string | null;
  reference_description: string | null;
  tuning_intent: "transparent" | "balanced" | "obvious";
  protected_ranges: { start: number; end: number }[];
}

export interface TuningControls {
  key: string | null;
  scale: "major" | "minor" | "chromatic" | null;
  strength: number;
  retune_ms: number;
  max_correction_cents: number;
  vibrato_preservation: number;
  formant_shift: number;
  breath_protection: number;
  eq_amount: number;
  deesser_amount: number;
  compression_amount: number;
  ambience_amount: number;
  vocal_gain_db: number;
}

export interface TuningPlan {
  rationale: string[];
  controls: TuningControls;
  notes: NoteEvent[];
  chords: ChordEvent[];
  protected_ranges: { start: number; end: number }[];
  source: string;
}

export interface OutputFile {
  kind: string;
  filename: string;
  media_type: string;
}

export interface Project {
  id: string;
  name: string;
  mode: InputMode;
  status: string;
  created_at: string;
  updated_at: string;
  has_report: boolean;
  has_outputs: boolean;
  report: AnalysisReport | null;
  brief: CreativeBrief | null;
  plan: TuningPlan | null;
  outputs: OutputFile[];
}

export interface Job {
  id: string;
  project_id: string;
  kind: "analysis" | "preview" | "render";
  stage: string;
  progress: number;
  status: "queued" | "running" | "complete" | "failed" | "cancelled";
  message: string;
  error: string | null;
}

export interface SystemInfo {
  models: Record<string, string | number | boolean | null>;
  rating_count: number;
  model_budget_gb: number;
  llm_config: { endpoint: string | null; model: string | null };
  keep_masters: boolean;
}
