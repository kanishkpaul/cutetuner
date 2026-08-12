import type { CreativeBrief, Job, Project, SystemInfo, TuningControls } from "./types";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let message = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      message = body.detail ?? message;
    } catch {
      // Keep the HTTP message when the response is not JSON.
    }
    throw new Error(message);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export const api = {
  projects: () => request<Project[]>("/api/projects"),
  project: (id: string) => request<Project>(`/api/projects/${id}`),
  system: () => request<SystemInfo>("/api/system"),
  configureLlm: (endpoint: string | null, model: string | null) => request("/api/settings/llm", {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ endpoint, model }),
  }),
  keepMasters: (keep: boolean) => request(`/api/settings/masters?keep=${keep}`, { method: "PUT" }),
  create: (form: FormData) => request<Project>("/api/projects", { method: "POST", body: form }),
  remove: (id: string) => request<void>(`/api/projects/${id}`, { method: "DELETE" }),
  removeAll: () => request<void>("/api/projects", { method: "DELETE" }),
  analyze: (id: string) => request<Job>(`/api/projects/${id}/analyze`, { method: "POST" }),
  brief: (id: string, brief: CreativeBrief) => request(`/api/projects/${id}/brief`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(brief),
  }),
  plan: (id: string, controls: TuningControls) => request(`/api/projects/${id}/plan`, {
    method: "PUT", headers: { "Content-Type": "application/json" }, body: JSON.stringify(controls),
  }),
  previews: (id: string) => request<Job>(`/api/projects/${id}/previews`, { method: "POST" }),
  render: (id: string, controls: TuningControls) => request<Job>(`/api/projects/${id}/render`, {
    method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ controls }),
  }),
  cancel: (id: string) => request<Job>(`/api/jobs/${id}/cancel`, { method: "POST" }),
  rate: (id: string, winner: string, comment = "") => request(`/api/projects/${id}/ratings`, {
    method: "POST", headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ winner, comment, context: {} }),
  }),
};
