import type { CompressionResult, Mode } from '../types';
const API = import.meta.env.VITE_API_URL || 'http://localhost:8000/api';
export async function compress(text: string, mode: Mode): Promise<CompressionResult> {
  const response = await fetch(`${API}/compress`, { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({text, mode}) });
  if (!response.ok) { const body = await response.json().catch(() => ({})); throw new Error(body.detail || 'Compression failed.'); }
  return response.json();
}
export async function upload(file: File, mode: Mode): Promise<CompressionResult> {
  const body = new FormData(); body.append('file', file); body.append('mode', mode);
  const response = await fetch(`${API}/upload`, { method: 'POST', body });
  if (!response.ok) { const data = await response.json().catch(() => ({})); throw new Error(data.detail || 'Upload failed.'); }
  return response.json();
}
